#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../csrc/firered_vad_stream_packed.h"
#include "frontend/wav.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <model.param> <model.bin> <cmvn_means> <cmvn_istd> <wav_file>\n", argv[0]);
        return 1;
    }
    
    const char* model_param = argv[1];
    const char* model_bin = argv[2];
    const char* cmvn_means = argv[3];
    const char* cmvn_istd = argv[4];
    const char* wav_file = argv[5];
    
    printf("FireRedVAD NCNN Stream Packed Cache C API Test\n");
    printf("==============================================\n\n");
    
    // 创建 VAD
    printf("Creating VAD (packed cache version)...\n");
    FireredVADHandle vad = firered_vad_create(model_param, model_bin, cmvn_means, cmvn_istd);
    if (!vad) {
        fprintf(stderr, "Failed to create VAD\n");
        return 1;
    }
    printf("VAD created successfully\n\n");
    
    // 读取音频
    printf("Loading audio: %s\n", wav_file);
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to load audio\n");
        firered_vad_destroy(vad);
        return 1;
    }
    
    printf("Audio loaded: %d samples, %d Hz, %.3f sec\n\n", 
           reader.num_samples(), reader.sample_rate(),
           (float)reader.num_samples() / reader.sample_rate());
    
    std::vector<float> audio_data = reader.GetMonoData();
    int num_samples = audio_data.size();
    int sample_rate = reader.sample_rate();
    
    // 处理音频（每次 10ms = 160 samples）
    printf("Processing audio (10ms chunks, packed cache)...\n");
    printf("---------------------------------------------\n");
    
    const int chunk_size = 160;  // 10ms @ 16kHz
    int num_chunks = (num_samples + chunk_size - 1) / chunk_size;
    
    int speech_frames = 0;
    int total_results = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int remaining = num_samples - offset;
        int process_size = (remaining < chunk_size) ? remaining : chunk_size;
        
        // 转换为 int16_t
        std::vector<int16_t> chunk_audio(process_size);
        for (int j = 0; j < process_size; j++) {
            chunk_audio[j] = (int16_t)audio_data[offset + j];
        }
        
        FireredVADResult result;
        if (firered_vad_process_stream(vad, chunk_audio.data(), process_size, &result) == 0) {
            total_results++;
            
            // 打印所有结果
            printf("Frame %4d: time=%.3fs, confidence=%.4f, %s\n",
                   result.frame_offset,
                   result.frame_offset * 0.01f,
                   result.confidence,
                   result.is_speech ? "SPEECH" : "silence");
            
            if (result.is_speech) {
                speech_frames++;
            }
        }
    }
    
    printf("---------------------------------------------\n");
    printf("Processing complete\n\n");
    
    printf("Statistics:\n");
    printf("  Total chunks: %d\n", num_chunks);
    printf("  Total results: %d\n", total_results);
    printf("  Speech detected: %d frames (%.1f%%)\n", 
           speech_frames, total_results > 0 ? (float)speech_frames / total_results * 100 : 0);
    printf("  Final frame offset: %d\n", firered_vad_get_frame_offset(vad));
    printf("  Audio duration: %.3f sec\n", (float)num_samples / sample_rate);
    
    // 清理
    firered_vad_destroy(vad);
    
    return 0;
}
