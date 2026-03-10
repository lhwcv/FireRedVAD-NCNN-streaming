# Quick Start Guide

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

Success output:
```
Frame    0: time=0.000s, confidence=0.1953, silence
Frame    1: time=0.010s, confidence=0.1413, silence
...
```

### 3. Four-Way Comparison

```bash
./test/compare_four.sh test/hello_zh.wav
```

View comparison plot:
```bash
# View locally
xdg-open output/compare_four.png

# Or copy to your location
cp output/compare_four.png /path/to/your/images/
```

## Complete Conversion Flow

### Prerequisites

- FireRedVAD original model (`pretrained_models/xukaituo/FireRedVAD/VAD/`)
- Python environment (torch, onnx, pnnx)

### Steps

```bash
# 1. Go to convert directory
cd convert

# 2. Export NCNN model
python3 export_ncnn_packed_cache.py

# 3. Verify NCNN model
python3 firered_vad_packed_cache_ncnn.py

# 4. Copy models to models/
cp firered_vad_packed_cache.ncnn.* ../models/

# 5. Convert CMVN parameters
python3 -c "
import numpy as np
import sys
sys.path.insert(0, '../../3rd/FireRedVAD')
from fireredvad.core.audio_feat import CMVN

cmvn = CMVN('../../3rd/FireRedVAD/pretrained_models/xukaituo/FireRedVAD/VAD/cmvn.ark')
cmvn.means.astype(np.float32).tofile('../models/cmvn_means.bin')
cmvn.inverse_std_variances.astype(np.float32).tofile('../models/cmvn_istd.bin')
"

# 6. Verify
cd ..
./build.sh test
```

## Integration Guide

### C++ Integration

```cpp
#include "firered_vad_stream_packed.h"

// Create
auto vad = firered_vad_create("model.param", "model.bin", "cmvn_m.bin", "cmvn_i.bin");

// Process
FireredVADResult result;
firered_vad_process_stream(vad, audio_chunk, 160, &result);

// Cleanup
firered_vad_destroy(vad);
```

### Build Options

```cmake
find_package(firered_vad REQUIRED)
target_link_libraries(your_target PRIVATE firered_vad::firered_vad_stream)
```

## Understanding the Differences

You may notice small differences between NCNN C++ and PyTorch outputs. This is **expected** and comes from:

**Feature Extraction (wenet frontend vs Kaldi)**:
- Single frame: Nearly identical (max diff < 0.0001)
- Streaming: Small accumulation errors (~0.045 mean diff)
- Speech detection: > 94% agreement

**The NCNN inference itself is accurate** - the difference is purely from feature extraction implementation details in streaming mode.

## Common Issues

**Q: NCNN not found?**
```bash
export NCNN_ROOT=/path/to/ncnn
./build.sh clean
./build.sh build
```

**Q: Test fails?**
```bash
# Check model files
ls -lh models/

# Re-convert
cd convert && python3 export_ncnn_packed_cache.py
```

**Q: Comparison plot not generated?**
```bash
# Install matplotlib
pip3 install matplotlib

# Run manually
python3 test/compare_four_streaming.py test/hello_zh.wav output
```

## Next Steps

- Read full README.md
- Check API documentation (src/c_api/firered_vad_stream_packed.h)
- Test with more audio files
- Integrate into your project
