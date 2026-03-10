# 模型文件说明

本目录存放 NCNN 模型和 CMVN 参数文件。

## 必需文件

- `firered_vad_packed_cache.ncnn.param` - NCNN 参数文件
- `firered_vad_packed_cache.ncnn.bin` - NCNN 权重文件
- `cmvn_means.bin` - CMVN 均值（80 个 float32）
- `cmvn_istd.bin` - CMVN 标准差倒数（80 个 float32）

## 如何获取

### 方法 1: 自行转换（推荐）

```bash
cd ../convert

# 1. 导出 NCNN 模型
python3 export_ncnn_packed_cache.py

# 2. 复制模型文件
cp firered_vad_packed_cache.ncnn.* ../models/

# 3. 转换 CMVN 参数
python3 -c "
import numpy as np
import sys
sys.path.insert(0, '../../3rd/FireRedVAD')
from fireredvad.core.audio_feat import CMVN

cmvn = CMVN('../../3rd/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD/cmvn.ark')
cmvn.means.astype(np.float32).tofile('../models/cmvn_means.bin')
cmvn.inverse_std_variances.astype(np.float32).tofile('../models/cmvn_istd.bin')
"
```

### 方法 2: 下载预转换模型

从项目 Release 页面下载预转换的模型文件。

## 验证模型

```bash
cd ..
./build.sh test
```

如果测试通过，说明模型文件正确。

## 文件大小参考

- `firered_vad_packed_cache.ncnn.param`: ~10 KB
- `firered_vad_packed_cache.ncnn.bin`: ~1.1 MB
- `cmvn_means.bin`: 320 bytes (80 * 4)
- `cmvn_istd.bin`: 320 bytes (80 * 4)
