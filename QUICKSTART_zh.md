# 快速开始指南

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

看到输出表示成功：
```
Frame    0: time=0.000s, confidence=0.1953, silence
Frame    1: time=0.010s, confidence=0.1413, silence
...
```

### 3. 四者对比

```bash
./test/compare_four.sh test/hello_zh.wav
```

查看对比图：
```bash
# 在本地查看
xdg-open output/compare_four.png

# 或复制到指定位置
cp output/compare_four.png /path/to/your/images/
```

## 完整转换流程

### 前提条件

- FireRedVAD 原始模型（`pretrained_models/xukaituo/FireRedVAD/VAD/`）
- Python 环境（torch, onnx, pnnx）

### 步骤

```bash
# 1. 进入转换目录
cd convert

# 2. 导出 NCNN 模型
python3 export_ncnn_packed_cache.py

# 3. 验证 NCNN 模型
python3 firered_vad_packed_cache_ncnn.py

# 4. 复制模型到 models/
cp firered_vad_packed_cache.ncnn.* ../models/

# 5. 转换 CMVN 参数
python3 -c "
import numpy as np
import sys
sys.path.insert(0, '../../3rd/FireRedVAD')
from fireredvad.core.audio_feat import CMVN

cmvn = CMVN('../../3rd/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD/cmvn.ark')
cmvn.means.astype(np.float32).tofile('../models/cmvn_means.bin')
cmvn.inverse_std_variances.astype(np.float32).tofile('../models/cmvn_istd.bin')
"

# 6. 验证
cd ..
./build.sh test
```

## 集成到你的项目

### C++ 集成

```cpp
#include "firered_vad_stream_packed.h"

// 创建
auto vad = firered_vad_create("model.param", "model.bin", "cmvn_m.bin", "cmvn_i.bin");

// 处理
FireredVADResult result;
firered_vad_process_stream(vad, audio_chunk, 160, &result);

// 清理
firered_vad_destroy(vad);
```

### 编译选项

```cmake
find_package(firered_vad REQUIRED)
target_link_libraries(your_target PRIVATE firered_vad::firered_vad_stream)
```

## 常见问题

**Q: 找不到 NCNN？**
```bash
export NCNN_ROOT=/path/to/ncnn
./build.sh clean
./build.sh build
```

**Q: 测试失败？**
```bash
# 检查模型文件
ls -lh models/

# 重新转换
cd convert && python3 export_ncnn_packed_cache.py
```

**Q: 对比图无法生成？**
```bash
# 安装 matplotlib
pip3 install matplotlib

# 手动运行
python3 test/compare_four_streaming.py test/hello_zh.wav output
```

## 下一步

- 阅读完整 README.md
- 查看 API 文档（src/c_api/firered_vad_stream_packed.h）
- 运行更多测试音频
- 集成到你的项目
