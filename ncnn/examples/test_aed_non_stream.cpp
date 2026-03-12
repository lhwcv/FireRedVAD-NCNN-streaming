#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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
    if ((int)means.size() < feat_dim || (int)istd.size() < feat_dim) return;
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
        printf("Usage: %s <aed.param> <aed.bin> <cmvn_means_aed.bin> <cmvn_istd_aed.bin> <wav_file>\n", argv[0]);
        return 1;
    }

    const char* model_param = argv[1];
    const char* model_bin   = argv[2];
    const char* cmvn_means  = argv[3];
    const char* cmvn_istd   = argv[4];
    const char* wav_file    = argv[5];

    // Load wav
    vad::WavReader reader;
    if (!reader.Open(wav_file))
    {
        fprintf(stderr, "Failed to open wav: %s\n", wav_file);
        return 1;
    }
    std::vector<float> mono = reader.GetMonoData();
    int sample_rate = reader.sample_rate();
    if (sample_rate != 16000)
    {
        fprintf(stderr, "Warning: sample rate is %d, expected 16000\n", sample_rate);
    }

    // Fbank parameters
    const int feat_dim = 80;
    const int frame_len = 400;   // 25ms
    const int frame_shift = 160; // 10ms

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
        fprintf(stderr, "Failed to read cmvn means: %s\n", cmvn_means);
        return 1;
    }
    if (!load_binary_vector(cmvn_istd, istd))
    {
        fprintf(stderr, "Failed to read cmvn istd: %s\n", cmvn_istd);
        return 1;
    }
    apply_cmvn(means, istd, features, num_frames, feat_dim);

    // NCNN inference
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

    // Expect dims=2 with width=3 (classes), height=T (frames)
    int T = 0;
    if (out_probs.dims == 2 && out_probs.w == 3)
    {
        T = out_probs.h;
        for (int t = 0; t < T; ++t)
        {
            const float* p = out_probs.row(t);
            float speech  = p[0];
            float singing = p[1];
            float music   = p[2];
            printf("Frame %4d: time=%.3fs, speech=%.4f, singing=%.4f, music=%.4f\n",
                   t, t * 0.01f, speech, singing, music);
        }
    }
    else if (out_probs.dims == 2 && out_probs.h == 3)
    {
        // Transposed layout: height=3, width=T
        T = out_probs.w;
        for (int t = 0; t < T; ++t)
        {
            const float* row0 = out_probs.row(0);
            const float* row1 = out_probs.row(1);
            const float* row2 = out_probs.row(2);
            float speech  = row0[t];
            float singing = row1[t];
            float music   = row2[t];
            printf("Frame %4d: time=%.3fs, speech=%.4f, singing=%.4f, music=%.4f\n",
                   t, t * 0.01f, speech, singing, music);
        }
    }
    else
    {
        fprintf(stderr, "Unexpected output dims=%d w=%d h=%d c=%d\n", out_probs.dims, out_probs.w, out_probs.h, out_probs.c);
        return 1;
    }

    return 0;
}

