# FireRedVAD NCNN 部署模块 (Stream-VAD)

**[中文](README_zh.md)** | **[English](README.md)**

简洁高效的 FireRedVAD 语音活动检测（VAD）NCNN 部署实现，支持流式推理。

**✨ 默认模型：官方 Stream-VAD** - FireRedVAD 团队专用流式模型

## 特性

- ✅ **流式推理**: 支持 10ms 帧处理，25ms 窗口延迟
- ✅ **NCNN 优化**: 使用腾讯 NCNN 框架，轻量高效
- ✅ **打包 Cache**: 解决 NCNN 多输入限制，8 个 cache 打包为 1 个
- ✅ **无第三方依赖**: 使用 wenet frontend，无需 kaldi-native-fbank
- ✅ **CMVN 归一化**: 与 Kaldi 特征一致
- ✅ **C API**: 简洁的 C 接口，易于集成
- ✅ **高精度**: NCNN Python 与 PyTorch 差异 < 0.001，NCNN C++ 语音帧一致性 97.4%
- ✅ **Stream-VAD 模型**: 官方流式专用模型（非 VAD 模型适配流式）
- TODO: implement post filter [stream_vad_postprocessor](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/core/stream_vad_postprocessor.py)

![Four-way Comparison](output_stream/compare_four_stream_hello_zh.png)


## 目录结构

```
VAD_slim/
├── README.md              # 英文文档
├── README_zh.md           # 中文文档（本文件）
├── QUICKSTART.md          # 快速开始指南（英文）
├── QUICKSTART_zh.md       # 快速开始指南（中文）
├── STREAM_VAD_COMPARISON.md  # Stream-VAD 对比报告
├── build.sh               # 一键构建脚本
├── CMakeLists.txt         # CMake 配置
├── convert/               # 模型转换脚本
│   ├── export_ncnn_packed_cache_stream.py  # 导出 Stream-VAD NCNN 模型
│   └── firered_vad_packed_cache_stream_ncnn.py  # 验证 Stream-VAD 模型
├── src/                   # 源代码
│   ├── c_api/             # C API 实现
│   │   ├── firered_vad_stream_packed.h
│   │   ├── firered_vad_stream_packed.cpp
│   │   └── test_stream_packed.cpp
│   └── frontend/          # 特征提取
│       ├── fbank.h        # Fbank 特征（流式带状态）
│       ├── fft.h          # FFT 实现
│       ├── fft.cc
│       └── wav.h          # WAV 读取
├── test/                  # 测试脚本
│   ├── compare_four_streaming_stream.py  # 四者对比（Stream-VAD）
│   └── compare_with_post.py              # 后处理对比
├── models/                # 模型文件（Stream-VAD）
│   ├── firered_vad_packed_cache_stream.ncnn.param
│   ├── firered_vad_packed_cache_stream.ncnn.bin
│   ├── cmvn_means_stream.bin
│   └── cmvn_istd_stream.bin
└── output_stream/         # 测试输出（Stream-VAD）
    └── compare_four_stream_*.png  # 四者对比图
```

## 快速开始

### 1. 环境准备

```bash
# 系统依赖
sudo apt-get install -y cmake g++ libopenmp-dev

# NCNN（已预编译）
NCNN_ROOT=/opt/ncnn-20260113

# Python 依赖（用于模型转换）
pip3 install torch onnx pnnx soundfile matplotlib
```

### 2. 构建

```bash
# 一键构建（推荐）
./build.sh build

# 或分步构建
./build.sh build
./build.sh install
./build.sh test
```

### 3. 测试

```bash
# 运行测试（使用 hello_zh.wav，Stream-VAD 模型）
./build.sh test

# 运行四者对比（PyTorch/ONNX/NCNN Python/NCNN C++）
cd test
python3 compare_four_streaming_stream.py

# 查看对比图
# output_stream/compare_four_stream_hello_zh.png
```

## API 使用

### C API

```c
#include <firered_vad/firered_vad_stream_packed.h>

// 1. 创建 VAD 实例（Stream-VAD 模型）
FireredVADHandle vad = firered_vad_create(
    "firered_vad_packed_cache_stream.ncnn.param",
    "firered_vad_packed_cache_stream.ncnn.bin",
    "cmvn_means_stream.bin",
    "cmvn_istd_stream.bin"
);

// 2. 处理音频块（每块 10ms，160 采样点 @ 16kHz）
FireredVADResult result;
for (int i = 0; i < num_chunks; i++) {
    firered_vad_process_stream(vad, audio_chunk, chunk_size, &result);
    printf("Frame %d: confidence=%.4f, is_speech=%d\n", 
           result.frame_offset, result.confidence, result.is_speech);
}

// 3. 清理
firered_vad_destroy(vad);
```

### Python API（通过 ctypes）

```python
import ctypes

# 加载库
lib = ctypes.CDLL("./build/libfirered_vad_stream.so")

# 创建 VAD
vad = lib.firered_vad_create(
    b"models/firered_vad_packed_cache_stream.ncnn.param",
    b"models/firered_vad_packed_cache_stream.ncnn.bin",
    b"models/cmvn_means_stream.bin",
    b"models/cmvn_istd_stream.bin"
)

# 处理音频...
# 完整示例见 test/compare_four_streaming_stream.py
```

## 模型转换

### 转换 Stream-VAD 模型

```bash
cd convert

# 1. 导出带打包 cache 的 ONNX（Stream-VAD 权重）
python3 export_ncnn_packed_cache_stream.py

# 输出:
#   - firered_vad_packed_cache_stream.onnx
#   - firered_vad_packed_cache_stream.onnx.data

# 2. 使用 PNNX 转换为 NCNN
pnnx firered_vad_packed_cache_stream.onnx inputshape=[1,1,80]

# 3. 移动模型文件
mv firered_vad_packed_cache_stream.ncnn.param ../models/
mv firered_vad_packed_cache_stream.ncnn.bin ../models/

# 4. 准备 CMVN（从 FireRedVAD Stream-VAD 预训练模型）
# 详见 export_ncnn_packed_cache_stream.py
```

### 模型文件

所有模型文件已预转换并放在 `models/` 目录：

- `firered_vad_packed_cache_stream.ncnn.param` - Stream-VAD 网络结构
- `firered_vad_packed_cache_stream.ncnn.bin` - Stream-VAD 权重 (1.1MB)
- `cmvn_means_stream.bin` - CMVN 均值（Stream-VAD）
- `cmvn_istd_stream.bin` - CMVN 标准差倒数（Stream-VAD）

## 精度验证

### 四者对比（Stream-VAD）

| 对比 | 最大差异 | 平均差异 | 语音帧一致性 |
|------|----------|----------|-------------|
| PyTorch vs ONNX | 0.000000 | 0.000000 | 100.0% |
| PyTorch vs NCNN Python | 0.000723 | 0.000136 | 100.0% |
| PyTorch vs NCNN C++ | 0.601691 | 0.026116 | 97.4% |

**关键发现：**
- ✅ ONNX 与 PyTorch 完全一致（零差异）
- ✅ NCNN Python 精度优秀（最大差异 < 0.001）
- ✅ NCNN C++ 实用精度（97.4% 一致性，语音段偏差 < 20ms）
- ⚠️ C++ 差异来自 frontend 特征提取（wenet C++ vs Python），非 NCNN 推理问题

详见 `STREAM_VAD_COMPARISON.md` 分析报告。

## 流式推理

### 参数

- **帧移**: 10ms (160 采样点 @ 16kHz)
- **帧长**: 25ms (400 采样点 @ 16kHz)
- **延迟**: 25ms (一帧)
- **Cache**: 8 个 DFSMN cache，打包为 [1, 1024, 19]

### 实时处理

```c++
// 实时处理音频（10ms 块）
while (recording) {
    int16_t chunk[160];  // 10ms @ 16kHz
    record_audio(chunk, 160);
    
    FireredVADResult result;
    firered_vad_process_stream(vad, chunk, 160, &result);
    
    if (result.is_speech) {
        // 检测到语音
    }
}
```

## 集成方式

### CMake

```cmake
find_library(FIREREDVAD_LIB firered_vad_stream PATHS /path/to/VAD_slim/build)
include_directories(/path/to/VAD_slim/src/c_api)

target_link_libraries(your_app ${FIREREDVAD_LIB})
```

### Makefile

```makefile
FIREREDVAD_LIB = /path/to/VAD_slim/build/libfirered_vad_stream.so
FIREREDVAD_INC = /path/to/VAD_slim/src/c_api

your_app: your_app.cpp
    g++ -I$(FIREREDVAD_INC) -L$(dir $(FIREREDVAD_LIB)) \
        -o $@ $< -lfirered_vad_stream -lncnn
```

## 许可证

本项目用于教育和研究目的。请参考原始 [FireRedVAD 仓库](https://github.com/FireRedTeam/FireRedVAD) 了解许可条款。

## 致谢

- **FireRedVAD 团队**: 原始 VAD 模型和 Stream-VAD 模型
- **腾讯 NCNN**: 高性能神经网络推理框架
- **wenet**: Frontend 特征提取实现


## 参考


- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [NCNN](https://github.com/Tencent/ncnn)
- [PNNX](https://github.com/pnnx/pnnx)
- [WeNet](https://github.com/wenet-e2e/wenet)

