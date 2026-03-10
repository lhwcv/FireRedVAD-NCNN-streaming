# FireRedVAD NCNN Deployment Module

**[中文](README_zh.md)** | **[English](README.md)**

A clean and efficient FireRedVAD Voice Activity Detection (VAD) NCNN deployment implementation with streaming inference support.

## Features

- ✅ **Streaming Inference**: 10ms frame processing with 25ms window latency
- ✅ **NCNN Optimized**: Uses Tencent NCNN framework, lightweight and efficient
- ✅ **Packed Cache**: Solves NCNN multi-input limitation, 8 caches packed into 1
- ✅ **No Third-party Dependencies**: Uses wenet frontend, no kaldi-native-fbank required
- ✅ **CMVN Normalization**: Consistent with Kaldi features
- ✅ **C API**: Clean C interface, easy to integrate
- ✅ **High Accuracy**: Difference from PyTorch < 0.003 (NCNN Python)
- TODO: implement post filter [stream_vad_postprocessor](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/core/stream_vad_postprocessor.py)

## Directory Structure

```
VAD_slim/
├── README.md              # This document (English)
├── README_zh.md           # Chinese version
├── QUICKSTART.md          # Quick start guide (English)
├── QUICKSTART_zh.md       # Chinese version
├── build.sh               # One-click build script
├── CMakeLists.txt         # CMake configuration
├── convert/               # Model conversion scripts
│   ├── export_ncnn_packed_cache.py    # Export NCNN model
│   └── firered_vad_packed_cache_ncnn.py  # Verify NCNN model
├── src/                   # Source code
│   ├── c_api/             # C API implementation
│   │   ├── firered_vad_stream_packed.h
│   │   ├── firered_vad_stream_packed.cpp
│   │   └── test_stream_packed.cpp
│   └── frontend/          # Feature extraction
│       ├── fbank.h        # Fbank features
│       ├── fft.h          # FFT implementation
│       ├── fft.cc
│       └── wav.h          # WAV reader
├── test/                  # Test scripts
│   ├── compare_four.sh              # Four-way comparison shell
│   └── compare_four_streaming.py    # Four-way comparison Python
├── models/                # Model files (convert or download)
│   ├── firered_vad_packed_cache.ncnn.param
│   ├── firered_vad_packed_cache.ncnn.bin
│   ├── cmvn_means.bin
│   └── cmvn_istd.bin
└── output/                # Test output
```

## Quick Start

### 1. Environment Setup

```bash
# System dependencies
sudo apt-get install -y cmake g++ libopenmp-dev

# NCNN (pre-compiled)
NCNN_ROOT=/opt/ncnn-20260113

# Python dependencies (for model conversion)
pip3 install torch onnx pnnx soundfile
```

### 2. Build

```bash
# One-click build (recommended)
./build.sh build

# Or step-by-step
./build.sh build
./build.sh install
./build.sh test
```

### 3. Test

```bash
# Run test (using hello_zh.wav)
./build.sh test

# Run four-way comparison
cd test
./compare_four.sh /path/to/hello_zh.wav

# View comparison plot
# output/compare_four.png
```

## API Usage

### C API

```c
#include <firered_vad/firered_vad_stream_packed.h>

// 1. Create VAD instance
FireredVADHandle vad = firered_vad_create(
    "models/firered_vad_packed_cache.ncnn.param",
    "models/firered_vad_packed_cache.ncnn.bin",
    "models/cmvn_means.bin",
    "models/cmvn_istd.bin"
);

// 2. Process audio (10ms = 160 samples @ 16kHz)
int16_t audio_chunk[160];
FireredVADResult result;

while (has_audio) {
    read_audio(audio_chunk, 160);
    
    if (firered_vad_process_stream(vad, audio_chunk, 160, &result) == 0) {
        if (result.is_speech) {
            printf("Speech: %.4f\n", result.confidence);
        }
    }
}

// 3. Reset (for new audio)
firered_vad_reset(vad);

// 4. Destroy
firered_vad_destroy(vad);
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16000 Hz | Fixed |
| Frame Length | 25ms | 400 samples |
| Frame Shift | 10ms | 160 samples |
| Latency | 25ms | First frame |
| Feature Dim | 80 | Fbank |
| Cache Dim | [1, 1024, 19] | Packed 8 caches |

## Accuracy Comparison

| Version | Max Diff | Mean Diff | Speech Frame Agreement |
|---------|----------|-----------|----------------------|
| PyTorch vs ONNX | 0.000000 | 0.000000 | 100.0% |
| PyTorch vs NCNN Python | 0.002129 | 0.000458 | 100.0% |
| PyTorch vs NCNN C++ | 0.325093 | 0.045621 | 94.8% (wenet frontend vs kaldi diff) |

**Four-way Comparison Result**:

![Four-way Comparison](output/compare_four.png)

**Notes**:
- PyTorch/ONNX/NCNN Python are almost perfectly consistent
- NCNN C++ has slight differences due to **streaming feature extraction** (wenet frontend vs Kaldi), but acceptable for practical use
- The difference comes from **feature extraction accumulation errors** in streaming processing, not from NCNN inference itself
- Speech frame detection consistency > 94% (the difference is only from frontend feature extraction, NCNN inference is accurate)

### Why the Difference?

The NCNN C++ version uses **wenet frontend** for feature extraction to avoid third-party dependencies (kaldi-native-fbank). While the features are nearly identical (max diff < 0.0001 for single-frame extraction), the **streaming processing** introduces small accumulation differences:

1. **Pre-emphasis state management**: Streaming maintains state across frames
2. **Window boundary handling**: Slight differences in overlapping regions
3. **DC offset removal**: Per-frame mean calculation vs batch


## Model Conversion

### Step 1: Export ONNX

```python
# export_ncnn_packed_cache.py
class FireRedVADPackedCache(nn.Module):
    def forward(self, feat, caches_packed):
        # Split: [1, 1024, 19] -> 8 x [1, 128, 19]
        caches = list(caches_packed.chunk(8, dim=1))
        
        # Forward pass
        probs, new_caches = self.model.forward(feat, caches=caches)
        
        # Pack: 8 x [1, 128, 19] -> [1, 1024, 19]
        new_caches_packed = torch.cat(new_caches, dim=1)
        
        return probs, new_caches_packed
```

### Step 2: PNNX Conversion

```bash
pnnx firered_vad_packed_cache.onnx
# Output: firered_vad_packed_cache.ncnn.param/bin
```

### Step 3: Verify

```bash
# Python verification
python3 firered_vad_packed_cache_ncnn.py

# C++ verification
./build/test_vad_stream models/*.param models/*.bin models/cmvn_*.bin test/hello_zh.wav
```



## FAQ

### Q1: NCNN extract returns -100 error?

**A**: Check Mat creation:
```cpp
// ✅ Correct
cache_packed = ncnn::Mat(19, 1024);

// ❌ Wrong
cache_packed = ncnn::Mat(19, 1024, 1);
```

### Q2: Feature extraction inconsistent with Kaldi?

**A**: Ensure:
1. CMVN parameters correctly converted from Kaldi ark
2. Pre-emphasis uses streaming state management
3. Window function and FFT parameters match

### Q3: What is the streaming latency?

**A**: 
- First frame: 25ms (frame length)
- Subsequent frames: Output every 10ms
- Average latency: ~30ms

### Q4: How to integrate into my project?

**A**: 
1. Copy `src/c_api/` and `src/frontend/` to your project
2. Link NCNN library and OpenMP
3. Call C API

## License

This module is based on the FireRedVAD original model and follows its open source license.

## References

- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [NCNN](https://github.com/Tencent/ncnn)
- [PNNX](https://github.com/pnnx/pnnx)
- [WeNet](https://github.com/wenet-e2e/wenet)

## Contact

For issues, please open an Issue or contact the maintainers.
