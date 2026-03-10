# Model Files

This directory contains NCNN model and CMVN parameter files.

## Required Files

- `firered_vad_packed_cache.ncnn.param` - NCNN parameter file
- `firered_vad_packed_cache.ncnn.bin` - NCNN weights file
- `cmvn_means.bin` - CMVN means (80 float32 values)
- `cmvn_istd.bin` - CMVN inverse standard deviations (80 float32 values)

## How to Obtain

### Method 1: Convert Yourself (Recommended)

```bash
cd ../convert

# 1. Export NCNN model
python3 export_ncnn_packed_cache.py

# 2. Copy model files
cp firered_vad_packed_cache.ncnn.* ../models/

# 3. Convert CMVN parameters
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

### Method 2: Download Pre-converted Models

Download pre-converted model files from the project Release page.

## Verify Models

```bash
cd ..
./build.sh test
```

If the test passes, the model files are correct.

## File Size Reference

- `firered_vad_packed_cache.ncnn.param`: ~10 KB
- `firered_vad_packed_cache.ncnn.bin`: ~1.1 MB
- `cmvn_means.bin`: 320 bytes (80 * 4)
- `cmvn_istd.bin`: 320 bytes (80 * 4)
