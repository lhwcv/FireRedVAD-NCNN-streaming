#include "firered_vad_stream_packed.h"
#include "frontend/fbank.h"
#include "net.h"
#include <cstring>
#include <cstdio>
#include <vector>
#include <deque>

// 打包 cache 版本
#define CACHE_SIZE 1024  // 8 * 128
#define CACHE_LEN 19
#define FEAT_DIM 80

// 内部结构体
struct FireredVAD {
    // NCNN Net
    ncnn::Net* net;
    
    // 参数
    int sample_rate;
    int frame_shift_ms;       // 10ms
    int frame_length_ms;      // 25ms
    int feat_dim;
    float threshold;
    
    // 音频缓冲区（滑动窗口，保存最近 25ms）
    std::deque<float> audio_buffer;
    int frame_shift_samples;   // 160 samples = 10ms @ 16kHz
    int frame_length_samples;  // 400 samples = 25ms @ 16kHz
    
    // 打包 cache [1, 1024, 19]
    ncnn::Mat cache_packed;
    
    // 当前状态
    int frame_offset;
    
    // CMVN
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;
    
    // Fbank 计算器
    vad::Fbank* fbank;
};

extern "C" {

FireredVADHandle firered_vad_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd
) {
    FireredVAD* vad = new FireredVAD();
    if (!vad) {
        return NULL;
    }
    
    // 初始化参数
    vad->sample_rate = 16000;
    vad->frame_shift_ms = 10;
    vad->frame_length_ms = 25;
    vad->feat_dim = FEAT_DIM;
    vad->threshold = 0.5f;
    vad->frame_shift_samples = vad->sample_rate * vad->frame_shift_ms / 1000;
    vad->frame_length_samples = vad->sample_rate * vad->frame_length_ms / 1000;
    vad->frame_offset = 0;
    vad->use_cmvn = false;
    vad->net = NULL;
    vad->fbank = NULL;
    
    // 初始化打包 cache [w=19, h=1024, c=1] 对应 [1, 1024, 19]
    // 注意：使用 2 参数构造函数，避免 3 参数版本的潜在问题
    vad->cache_packed = ncnn::Mat(CACHE_LEN, CACHE_SIZE);
    vad->cache_packed.fill(0.0f);
    
    // 创建 NCNN Net
    vad->net = new ncnn::Net();
    if (!vad->net) {
        delete vad;
        return NULL;
    }
    
    // 加载模型
    if (vad->net->load_param(model_param) != 0) {
        fprintf(stderr, "Failed to load param: %s\n", model_param);
        delete vad->net;
        delete vad;
        return NULL;
    }
    
    if (vad->net->load_model(model_bin) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_bin);
        delete vad->net;
        delete vad;
        return NULL;
    }
    
    // 创建 Fbank 计算器
    vad->fbank = new vad::Fbank(vad->feat_dim, vad->sample_rate, 
                                 vad->frame_length_samples, vad->frame_shift_samples);
    if (!vad->fbank) {
        delete vad->net;
        delete vad;
        return NULL;
    }
    
    // 加载 CMVN
    if (cmvn_means && cmvn_istd) {
        FILE* fp = fopen(cmvn_means, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            int dim = size / sizeof(float);
            vad->cmvn_means.resize(dim);
            fread(vad->cmvn_means.data(), sizeof(float), dim, fp);
            fclose(fp);
            
            fp = fopen(cmvn_istd, "rb");
            if (fp) {
                vad->cmvn_istd.resize(dim);
                fread(vad->cmvn_istd.data(), sizeof(float), dim, fp);
                fclose(fp);
                vad->use_cmvn = true;
            }
        }
    }
    
    return vad;
}

void firered_vad_destroy(FireredVADHandle handle) {
    if (!handle) return;
    
    FireredVAD* vad = (FireredVAD*)handle;
    delete vad->net;
    delete vad->fbank;
    delete vad;
}

static void apply_cmvn(FireredVAD* vad, float* features, int num_frames) {
    if (!vad->use_cmvn || vad->cmvn_means.empty() || vad->cmvn_istd.empty()) {
        return;
    }
    
    for (int t = 0; t < num_frames; t++) {
        for (int d = 0; d < vad->feat_dim; d++) {
            int idx = t * vad->feat_dim + d;
            features[idx] = (features[idx] - vad->cmvn_means[d]) * vad->cmvn_istd[d];
        }
    }
}

int firered_vad_process_stream(
    FireredVADHandle handle,
    const int16_t* audio_data,
    int num_samples,
    FireredVADResult* result
) {
    if (!handle || !audio_data || !result) {
        return -1;
    }
    
    FireredVAD* vad = (FireredVAD*)handle;
    
    // 将 16bit PCM 转换为 float 并加入缓冲区
    for (int i = 0; i < num_samples; i++) {
        vad->audio_buffer.push_back((float)audio_data[i]);
    }
    
    // 保持缓冲区大小不超过 frame_length_samples (400 samples = 25ms)
    while ((int)vad->audio_buffer.size() > vad->frame_length_samples) {
        vad->audio_buffer.pop_front();
    }
    
    // 检查是否有足够的数据进行特征提取
    if ((int)vad->audio_buffer.size() < vad->frame_length_samples) {
        // 数据不足，返回默认值
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = vad->frame_offset;
        return 0;
    }
    
    // 提取当前帧的特征
    std::vector<float> frame_audio(vad->audio_buffer.begin(),
                                    vad->audio_buffer.end());
    
    std::vector<float> features;
    int num_frames = vad->fbank->Compute(frame_audio, &features);
    
    if (num_frames == 0 || features.empty()) {
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = vad->frame_offset;
        return 0;
    }
    
    // 应用 CMVN（只对最后一帧）
    if (vad->use_cmvn) {
        apply_cmvn(vad, features.data(), num_frames);
    }
    
    // 取最后一帧的特征
    std::vector<float> current_feat(vad->feat_dim);
    memcpy(current_feat.data(), 
           features.data() + (num_frames - 1) * vad->feat_dim,
           vad->feat_dim * sizeof(float));
    
    // 更新帧偏移
    vad->frame_offset++;
    
    // NCNN 推理（单帧，带打包 cache）
    // 输入：in0=feat[80, 1], in1=cache_packed[19, 1024]
    // 输出：out0=probs[1], out1=new_cache_packed[19, 1024]
    
    ncnn::Mat in_feat(vad->feat_dim, 1);
    memcpy(in_feat.data, current_feat.data(), vad->feat_dim * sizeof(float));
    
    // 使用 clone() 确保数据正确复制（与 Python 版本一致）
    ncnn::Mat in_feat_clone = in_feat.clone();
    ncnn::Mat cache_clone = vad->cache_packed.clone();
    
    ncnn::Extractor ex = vad->net->create_extractor();
    ex.input("in0", in_feat_clone);
    ex.input("in1", cache_clone);
    
    // 提取输出概率
    ncnn::Mat out_probs;
    int ret = ex.extract("out0", out_probs);
    
    if (ret != 0) {
        fprintf(stderr, "NCNN extract probs failed: ret=%d\n", ret);
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = vad->frame_offset;
        return 0;
    }
    
    // 提取新的 cache
    ncnn::Mat new_cache_packed;
    ret = ex.extract("out1", new_cache_packed);
    
    if (ret != 0) {
        fprintf(stderr, "NCNN extract cache failed: ret=%d\n", ret);
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = vad->frame_offset;
        return 0;
    }
    
    // 更新 cache
    vad->cache_packed = new_cache_packed;
    
    float confidence = out_probs[0];
    bool is_speech = confidence > vad->threshold;
    
    // 返回结果
    result->confidence = confidence;
    result->is_speech = is_speech;
    result->frame_offset = vad->frame_offset;
    return 0;
}

void firered_vad_reset(FireredVADHandle handle) {
    if (!handle) return;
    
    FireredVAD* vad = (FireredVAD*)handle;
    vad->audio_buffer.clear();
    vad->frame_offset = 0;
    
    // 重置 cache
    vad->cache_packed.fill(0.0f);
    
    // 重置 fbank pre-emphasis 状态
    if (vad->fbank) {
        vad->fbank->reset();
    }
}

int firered_vad_get_frame_offset(FireredVADHandle handle) {
    if (!handle) return 0;
    return ((FireredVAD*)handle)->frame_offset;
}

} // extern "C"
