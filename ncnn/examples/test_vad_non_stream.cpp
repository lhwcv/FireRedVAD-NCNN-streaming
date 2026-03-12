#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cmath>

#include "net.h"
#include "frontend/wav.h"
#include "frontend/fbank.h"

static bool load_binary_vector(const char* path, std::vector<float>& vec)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int dim = size / sizeof(float);
    vec.resize(dim);
    fread(vec.data(), sizeof(float), dim, fp);
    fclose(fp);
    return true;
}

static void apply_cmvn(const std::vector<float>& means,
                       const std::vector<float>& istd,
                       std::vector<float>& feat,
                       int num_frames, int feat_dim)
{
    if (means.size() < (size_t)feat_dim || istd.size() < (size_t)feat_dim) return;
    for (int t = 0; t < num_frames; ++t)
    {
        float* f = feat.data() + t * feat_dim;
        for (int d = 0; d < feat_dim; ++d)
        {
            f[d] = (f[d] - means[d]) * istd[d];
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        printf("Usage: %s <model.param> <model.bin> <cmvn_means_vad.bin> <cmvn_istd_vad.bin> <wav_file>\n", argv[0]);
        return 1;
    }

    const char* model_param = argv[1];
    const char* model_bin   = argv[2];
    const char* cmvn_means  = argv[3];
    const char* cmvn_istd   = argv[4];
    const char* wav_file    = argv[5];

    // Load audio
    vad::WavReader reader;
    if (!reader.Open(wav_file))
    {
        fprintf(stderr, "Failed to read wav: %s\n", wav_file);
        return 1;
    }

    std::vector<float> mono = reader.GetMonoData();
    int sample_rate = reader.sample_rate();
    if (sample_rate != 16000)
    {
        fprintf(stderr, "Warning: sample rate is %d, expected 16000\n", sample_rate);
    }

    // Extract fbank features for the whole audio
    const int feat_dim = 80;
    const int frame_len = 400;  // 25ms@16k
    const int frame_shift = 160; // 10ms@16k

    vad::Fbank fbank(feat_dim, 16000, frame_len, frame_shift);
    std::vector<float> features;
    int num_frames = fbank.Compute(mono, &features);
    if (num_frames <= 0)
    {
        fprintf(stderr, "No frames computed\n");
        return 1;
    }

    // CMVN
    std::vector<float> means, istd;
    if (!load_binary_vector(cmvn_means, means))
    {
        fprintf(stderr, "Failed to load cmvn means: %s\n", cmvn_means);
        return 1;
    }
    if (!load_binary_vector(cmvn_istd, istd))
    {
        fprintf(stderr, "Failed to load cmvn istd: %s\n", cmvn_istd);
        return 1;
    }
    apply_cmvn(means, istd, features, num_frames, feat_dim);

    // NCNN inference (non-stream, whole sequence)
    ncnn::Net net;
    if (net.load_param(model_param) != 0)
    {
        fprintf(stderr, "load_param failed: %s\n", model_param);
        return 1;
    }
    if (net.load_model(model_bin) != 0)
    {
        fprintf(stderr, "load_model failed: %s\n", model_bin);
        return 1;
    }

    // Prepare input: w=feat_dim, h=num_frames
    ncnn::Mat in_feat(feat_dim, num_frames);
    memcpy(in_feat.data, features.data(), sizeof(float) * features.size());
    ncnn::Mat in_feat_clone = in_feat.clone();

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in_feat_clone);

    ncnn::Mat out_probs;
    int ret = ex.extract("out0", out_probs);
    if (ret != 0)
    {
        fprintf(stderr, "NCNN extract failed: %d\n", ret);
        return 1;
    }

    // Print results per frame
    int T = num_frames;
    for (int i = 0; i < T; ++i)
    {
        float p = out_probs[i];
        bool is_speech = p > 0.5f;
        printf("Frame %4d: time=%.3fs, confidence=%.4f, %s\n",
               i, i * 0.01f, p, is_speech ? "SPEECH" : "silence");
    }

    return 0;
}

