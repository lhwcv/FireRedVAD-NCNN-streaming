# Quick Start Guide (Stream-VAD)

## 5-Minute Quick Test

### 1. Build

```bash
cd VAD_slim
./build.sh build
```

### 2. Test

```bash
./build.sh test
```

Success output (Stream-VAD model):
```
Frame    0: time=0.000s, confidence=0.0000, silence
Frame    1: time=0.010s, confidence=0.0383, silence
...
Frame   50: time=0.500s, confidence=0.8531, SPEECH
...
```

### 3. Four-Way Comparison

```bash
cd test
python3 compare_four_streaming_stream.py
```

View comparison plot:
```bash
# View locally
xdg-open output_stream/compare_four_stream_hello_zh.png

# Or copy to your location
cp output_stream/compare_four_stream_hello_zh.png /path/to/your/images/
```

## Complete Conversion Flow (Stream-VAD)

### Prerequisites

- FireRedVAD original model (`pretrained_models/xukaituo/FireRedVAD/Stream-VAD/`)
- Python environment (torch, onnx, pnnx)

### Steps

```bash
# 1. Go to convert directory
cd convert

# 2. Export Stream-VAD NCNN model
python3 export_ncnn_packed_cache_stream.py

# 3. Verify NCNN model
python3 firered_vad_packed_cache_stream_ncnn.py

# 4. Copy models to models/
cp firered_vad_packed_cache_stream.ncnn.* ../models/

# 5. Convert CMVN parameters (Stream-VAD)
python3 -c "
import numpy as np
import sys
sys.path.insert(0, '../../3rd/FireRedVAD')
from fireredvad.core.audio_feat import CMVN

cmvn = CMVN('../../3rd/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/Stream-VAD/cmvn.ark')
cmvn.means.astype(np.float32).tofile('../models/cmvn_means_stream.bin')
cmvn.inverse_std_variances.astype(np.float32).tofile('../models/cmvn_istd_stream.bin')
"

# 6. Verify
cd ..
./build.sh test
```

## Integration Guide

### C++ Integration

```cpp
#include "firered_vad_stream_packed.h"

// Create VAD (Stream-VAD model)
FireredVADHandle vad = firered_vad_create(
    "models/firered_vad_packed_cache_stream.ncnn.param",
    "models/firered_vad_packed_cache_stream.ncnn.bin",
    "models/cmvn_means_stream.bin",
    "models/cmvn_istd_stream.bin"
);

// Process audio (10ms chunks)
FireredVADResult result;
while (has_audio) {
    firered_vad_process_stream(vad, audio_chunk, 160, &result);
    if (result.is_speech) {
        // Speech detected
    }
}

// Cleanup
firered_vad_destroy(vad);
```

### Python Integration

```python
import ctypes
import numpy as np

# Load library
lib = ctypes.CDLL("./build/libfirered_vad_stream.so")

# Create VAD
vad = lib.firered_vad_create(
    b"models/firered_vad_packed_cache_stream.ncnn.param",
    b"models/firered_vad_packed_cache_stream.ncnn.bin",
    b"models/cmvn_means_stream.bin",
    b"models/cmvn_istd_stream.bin"
)

# Process audio
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

# Cleanup
lib.firered_vad_destroy(vad)
```

## Model Files

Pre-converted model files are available in `models/`:

- `firered_vad_packed_cache_stream.ncnn.param` - Network structure
- `firered_vad_packed_cache_stream.ncnn.bin` - Weights (1.1MB)
- `cmvn_means_stream.bin` - CMVN means
- `cmvn_istd_stream.bin` - CMVN inverse std

## Expected Results

### Single File Test

```bash
./build.sh test
```

Expected output for `hello_zh.wav` (2.32s):
- Speech detected: ~100 frames (43%)
- Speech segments: [(0.48, 1.85)]
- Processing time: < 100ms (real-time factor < 0.05)

### Four-Way Comparison

```bash
python3 test/compare_four_streaming_stream.py
```

Expected results:
- PyTorch vs ONNX: max=0.000000, mean=0.000000
- PyTorch vs NCNN Python: max=0.000723, mean=0.000136
- PyTorch vs NCNN C++: max=0.601691, mean=0.026116
- Speech frame consistency: 97.4%

## Troubleshooting

### Build Errors

```bash
# Missing NCNN
echo $NCNN_ROOT
# Should be: /opt/ncnn-20260113

# Missing dependencies
sudo apt-get install -y cmake g++ libopenmp-dev
```

### Model Not Found

```bash
# Check model files
ls -la models/*.ncnn.*
# Should show param and bin files

# Re-convert if needed
cd convert
python3 export_ncnn_packed_cache_stream.py
```

### Test Audio Not Found

```bash
# Provide your own test audio (16kHz, mono)
cp /path/to/your/audio.wav test/hello_zh.wav

# Or update build.sh to point to your audio
```

## Next Steps

1. **Test with your audio**: Replace `hello_zh.wav` with your test files
2. **Integrate into your project**: Use C API or Python wrapper
3. **Performance tuning**: Adjust threshold, window size in post-processing
4. **Deploy to edge**: Cross-compile for ARM/RISC-V platforms

For detailed documentation, see `README.md` or `README_zh.md`.
