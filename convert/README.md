# Model Conversion Scripts

## Default: Stream-VAD Model (Recommended)

**Use these scripts for Stream-VAD (official streaming-specialized model):**

```bash
# 1. Export ONNX with packed cache (Stream-VAD weights)
python3 export_ncnn_packed_cache_stream.py

# 2. Convert to NCNN using PNNX
pnnx firered_vad_packed_cache_stream.onnx inputshape=[1,1,80]

# 3. Move model files to models/
mv firered_vad_packed_cache_stream.ncnn.param ../models/
mv firered_vad_packed_cache_stream.ncnn.bin ../models/
```

**Output files:**
- `firered_vad_packed_cache_stream.onnx` - ONNX model
- `firered_vad_packed_cache_stream.ncnn.param` - NCNN network structure
- `firered_vad_packed_cache_stream.ncnn.bin` - NCNN weights (1.1MB)

## Legacy: VAD Model (Not Recommended)

The following scripts use the original VAD model (not Stream-VAD):

- `export_ncnn_packed_cache.py` - Export VAD model (not Stream-VAD)
- `firered_vad_packed_cache_ncnn.py` - Verify VAD model

**Note:** These are kept for reference only. Use Stream-VAD scripts for new deployments.

## CMVN Preparation

After converting the model, prepare CMVN parameters:

```bash
# For Stream-VAD
python3 -c "
import numpy as np
from fireredvad.core.audio_feat import CMVN

cmvn = CMVN('../../3rd/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/Stream-VAD/cmvn.ark')
cmvn.means.astype(np.float32).tofile('../models/cmvn_means_stream.bin')
cmvn.inverse_std_variances.astype(np.float32).tofile('../models/cmvn_istd_stream.bin')
"
```

## Model Comparison

| Model | Use Case | Recommendation |
|-------|----------|----------------|
| **Stream-VAD** | Streaming inference | ✅ **Recommended** |
| VAD (adapted) | Non-streaming or reference | ⚠️ Legacy |

Stream-VAD is specifically trained for streaming inference and provides better accuracy for real-time applications.
