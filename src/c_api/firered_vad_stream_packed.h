#ifndef FIRERED_VAD_STREAM_PACKED_H
#define FIRERED_VAD_STREAM_PACKED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// VAD 句柄
typedef struct FireredVAD* FireredVADHandle;

// VAD 结果
typedef struct {
    float confidence;          // 置信度 (0-1)
    bool is_speech;            // 是否语音
    int frame_offset;          // 帧偏移 (从0开始，每帧10ms)
} FireredVADResult;

/**
 * 创建 VAD 实例（NCNN 打包 cache 版本）
 * @param model_param NCNN 参数文件路径
 * @param model_bin NCNN 模型文件路径
 * @param cmvn_means CMVN 均值文件路径
 * @param cmvn_istd CMVN 标准差倒数文件路径
 * @return VAD 句柄，失败返回 NULL
 */
FireredVADHandle firered_vad_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd
);

/**
 * 销毁 VAD 实例
 */
void firered_vad_destroy(FireredVADHandle handle);

/**
 * 处理音频数据（流式，单帧，带打包 cache）
 * 
 * 外部每次送入 10ms (160 samples @ 16kHz) 的 16bit PCM 音频
 * 内部使用滑动窗口：帧长25ms，帧移10ms
 * 使用打包 cache [1, 1024, 19] 保持上下文
 * 
 * @param handle VAD 句柄
 * @param audio_data 16bit PCM 音频数据（建议每次160 samples = 10ms）
 * @param num_samples 采样点数（建议160）
 * @param result 输出结果
 * @return 0 成功，-1 失败
 */
int firered_vad_process_stream(
    FireredVADHandle handle,
    const int16_t* audio_data,
    int num_samples,
    FireredVADResult* result
);

/**
 * 重置 VAD 状态（包括打包 cache）
 */
void firered_vad_reset(FireredVADHandle handle);

/**
 * 获取当前帧偏移
 */
int firered_vad_get_frame_offset(FireredVADHandle handle);

#ifdef __cplusplus
}
#endif

#endif // FIRERED_VAD_STREAM_PACKED_H
