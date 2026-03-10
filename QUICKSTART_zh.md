# 快速开始指南 (Stream-VAD)

## 5 分钟快速测试

### 1. 构建

```bash
cd VAD_slim
./build.sh build
```

### 2. 测试

```bash
./build.sh test
```

看到输出表示成功（Stream-VAD 模型）：
```
Frame    0: time=0.000s, confidence=0.0000, silence
Frame    1: time=0.010s, confidence=0.0383, silence
...
Frame   50: time=0.500s, confidence=0.8531, SPEECH
...
```

### 3. 四者对比

```bash
cd test
python3 compare_four_streaming_stream.py
```

查看对比图：
```bash
# 在本地查看
xdg-open output_stream/compare_four_stream_hello_zh.png

# 或复制到指定位置
cp output_stream/compare_four_stream_hello_zh.png /path/to/your/images/
```

## 完整转换流程 (Stream-VAD)

### 前提条件

- FireRedVAD 原始模型（`pretrained_models/xukaituo/FireRedVAD/Stream-VAD/`）
- Python 环境（torch, onnx, pnnx）

### 步骤

```bash
# 1. 进入转换目录
cd convert

# 2. 导出 Stream-VAD NCNN 模型
python3 export_ncnn_packed_cache_stream.py

# 3. 验证 NCNN 模型
python3 firered_vad_packed_cache_stream_ncnn.py

# 4. 复制模型到 models/
cp firered_vad_packed_cache_stream.ncnn.* ../models/

# 5. 转换 CMVN 参数（Stream-VAD）
python3 -c "
import numpy as np
import sys
sys.path.insert(0, '../../3rd/FireRedVAD')
from fireredvad.core.audio_feat import CMVN

cmvn = CMVN('../../3rd/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/Stream-VAD/cmvn.ark')
cmvn.means.astype(np.float32).tofile('../models/cmvn_means_stream.bin')
cmvn.inverse_std_variances.astype(np.float32).tofile('../models/cmvn_istd_stream.bin')
"

# 6. 验证
cd ..
./build.sh test
```

## 集成到你的项目

### C++ 集成

```cpp
#include "firered_vad_stream_packed.h"

// 创建 VAD（Stream-VAD 模型）
FireredVADHandle vad = firered_vad_create(
    "models/firered_vad_packed_cache_stream.ncnn.param",
    "models/firered_vad_packed_cache_stream.ncnn.bin",
    "models/cmvn_means_stream.bin",
    "models/cmvn_istd_stream.bin"
);

// 处理音频（10ms 块）
FireredVADResult result;
while (has_audio) {
    firered_vad_process_stream(vad, audio_chunk, 160, &result);
    if (result.is_speech) {
        // 检测到语音
    }
}

// 清理
firered_vad_destroy(vad);
```

### Python 集成

```python
import ctypes
import numpy as np

# 加载库
lib = ctypes.CDLL("./build/libfirered_vad_stream.so")

# 创建 VAD
vad = lib.firered_vad_create(
    b"models/firered_vad_packed_cache_stream.ncnn.param",
    b"models/firered_vad_packed_cache_stream.ncnn.bin",
    b"models/cmvn_means_stream.bin",
    b"models/cmvn_istd_stream.bin"
)

# 处理音频
audio = np.frombuffer(audio_bytes, dtype=np.int16)
chunk_size = 160  # 10ms @ 16kHz

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    result = FireredVADResult()
    lib.firered_vad_process_stream(
        vad, 
        chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        len(chunk),
        ctypes.byref(result)
    )
    print(f"Frame {result.frame_offset}: {result.confidence:.4f}")

# 清理
lib.firered_vad_destroy(vad)
```

## 模型文件

预转换的模型文件已在 `models/` 目录中：

- `firered_vad_packed_cache_stream.ncnn.param` - 网络结构
- `firered_vad_packed_cache_stream.ncnn.bin` - 权重 (1.1MB)
- `cmvn_means_stream.bin` - CMVN 均值
- `cmvn_istd_stream.bin` - CMVN 标准差倒数

## 预期结果

### 单文件测试

```bash
./build.sh test
```

`hello_zh.wav` (2.32s) 的预期输出：
- 语音检测：~100 帧 (43%)
- 语音段：[(0.48, 1.85)]
- 处理时间：< 100ms (实时因子 < 0.05)

### 四者对比

```bash
python3 test/compare_four_streaming_stream.py
```

预期结果：
- PyTorch vs ONNX: max=0.000000, mean=0.000000
- PyTorch vs NCNN Python: max=0.000723, mean=0.000136
- PyTorch vs NCNN C++: max=0.601691, mean=0.026116
- 语音帧一致性：97.4%

## 故障排除

### 构建错误

```bash
# 检查 NCNN 路径
echo $NCNN_ROOT
# 应该输出：/opt/ncnn-20260113

# 安装依赖
sudo apt-get install -y cmake g++ libopenmp-dev
```

### 模型文件未找到

```bash
# 检查模型文件
ls -la models/*.ncnn.*
# 应该显示 param 和 bin 文件

# 如需重新转换
cd convert
python3 export_ncnn_packed_cache_stream.py
```

### 测试音频未找到

```bash
# 提供自己的测试音频（16kHz, 单声道）
cp /path/to/your/audio.wav test/hello_zh.wav

# 或更新 build.sh 指向你的音频文件
```

## 下一步

1. **测试你的音频**：用你的测试文件替换 `hello_zh.wav`
2. **集成到项目**：使用 C API 或 Python 包装器
3. **性能调优**：在后处理中调整阈值、窗口大小
4. **部署到边缘**：为 ARM/RISC-V 平台交叉编译

详细文档请查看 `README.md` 或 `README_zh.md`。
