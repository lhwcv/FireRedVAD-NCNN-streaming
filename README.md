# FireRedVAD NCNN Deployment Module (Stream-VAD)

### ✅ NCNN models download here: [Download Models](convert/out) <br/>

Now this project has moved into origin repo  as Runtime: https://github.com/FireRedTeam/FireRedVAD/runtime/

==========


A clean and efficient FireRedVAD Voice Activity Detection (VAD) NCNN deployment implementation with streaming inference support.

**✨ Default Model: Official Stream-VAD** - Specialized streaming model from FireRedVAD team

## Features

- ✅ **Streaming Inference**: 10ms frame processing with 25ms window latency
- ✅ **NCNN Optimized**: Uses Tencent NCNN framework, lightweight and efficient
- ✅ **Packed Cache**: Solves NCNN multi-input limitation, 8 caches packed into 1
- ✅ **No Third-party Dependencies**: Uses wenet frontend, no kaldi-native-fbank required
- ✅ **CMVN Normalization**: Consistent with Kaldi features
- ✅ **C API**: Clean C interface, easy to integrate
- ✅ **High Accuracy**: Difference from PyTorch < 0.003 (NCNN Python), 97.4% speech frame consistency (NCNN C++)
- ✅ **Stream-VAD Model**: Official streaming-specialized model (not VAD model adapted for streaming)
- TODO: implement post filter [stream_vad_postprocessor](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/core/stream_vad_postprocessor.py)

## Prerequisites
- CMake 3.10+, C++ toolchain (Clang/GCC)
- curl/wget + unzip
- Python 3.10+ to convert models in `../convert`
- Optional: Android NDK r21+ for cross-build

##  Convert Models
- From `./convert/` run:
  - `python export_packed_cache_stream_vad.py` (Stream VAD, packed cache)
  - `python export_non_stream_vad.py` (Non-stream VAD)
  - `python export_aed.py` (Non-stream AED, 3-class)
- Outputs are placed in `./convert/out/`.

 
## Build Runtime (Host)
- `./build.sh`
  - Artifacts are copied to `out/`:
    - `libfirered_vad_stream.*`
    - `test_vad_stream`
    - `test_vad_non_stream`
    - `test_aed_non_stream`

## Build Runtime (Android)
- Build NCNN for Android first (32-bit by default):
  - `./3rd/build_android.sh 20260113 $ANDROID_NDK`
- Cross-build runtime:
  - `./build_android.sh $ANDROID_NDK`
  - Artifacts are copied to `out_android/armeabi-v7a/`

## Run C++ Executables
- Stream VAD (packed cache):
  - `./out/test_vad_stream ../convert/out/firered_vad_packed_cache_stream.ncnn.param ../convert/out/firered_vad_packed_cache_stream.ncnn.bin ../convert/out/cmvn_means_stream.bin ../convert/out/cmvn_istd_stream.bin ../../assets/hello_en.wav`
- Non-stream VAD:
  - `./out/test_vad_non_stream ../convert/out/firered_vad_non_stream.ncnn.param ../convert/out/firered_vad_non_stream.ncnn.bin ../convert/out/cmvn_means_vad.bin ../convert/out/cmvn_istd_vad.bin ../../assets/hello_en.wav`
- Non-stream AED (3-class):
  - `./out/test_aed_non_stream ../convert/out/firered_aed_non_stream.ncnn.param ../convert/out/firered_aed_non_stream.ncnn.bin ../convert/out/cmvn_means_aed.bin ../convert/out/cmvn_istd_aed.bin ../../assets/event.wav`



## API Usage

### C API

```c
#include <firered_vad/firered_vad_stream_packed.h>

// 1. Create VAD instance (Stream-VAD model)
FireredVADHandle vad = firered_vad_create(
    "firered_vad_packed_cache_stream.ncnn.param",
    "firered_vad_packed_cache_stream.ncnn.bin",
    "cmvn_means_stream.bin",
    "cmvn_istd_stream.bin"
);

// 2. Process audio chunks (10ms each, 160 samples @ 16kHz)
FireredVADResult result;
for (int i = 0; i < num_chunks; i++) {
    firered_vad_process_stream(vad, audio_chunk, chunk_size, &result);
    printf("Frame %d: confidence=%.4f, is_speech=%d\n", 
           result.frame_offset, result.confidence, result.is_speech);
}

// 3. Cleanup
firered_vad_destroy(vad);
```
 
## License

This project is for educational and research purposes. Please refer to the original [FireRedVAD repository](https://github.com/FireRedTeam/FireRedVAD) for licensing terms.

## Acknowledgments

- **FireRedVAD Team**: Original VAD model and Stream-VAD model
- **Tencent NCNN**: High-performance neural network inference framework
- **wenet**: Frontend feature extraction implementation

## Links


- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [NCNN](https://github.com/Tencent/ncnn)
- [PNNX](https://github.com/pnnx/pnnx)
- [WeNet](https://github.com/wenet-e2e/wenet)

