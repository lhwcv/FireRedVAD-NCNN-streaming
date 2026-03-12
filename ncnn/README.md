# FireRedVAD NCNN Runtime

This directory contains the NCNN-based runtime, build scripts, and C++ examples for FireRedVAD.

### Contents
- 3rd-party helpers under `3rd/` to download/build NCNN
- `build.sh`: host build, collects artifacts into `out/`
- `build_android.sh`: Android cross-build, collects artifacts into `out_android/<abi>/`
- C++ examples under `examples/`
- C API header under `csrc/`


### Prerequisites
- CMake 3.10+, C++ toolchain (Clang/GCC)
- curl/wget + unzip
- Python 3.10+ to convert models in `../convert`
- Optional: Android NDK r21+ for cross-build

###  Convert Models
- From `runtime/convert/` run:
  - `python export_packed_cache_stream_vad.py` (Stream VAD, packed cache)
  - `python export_non_stream_vad.py` (Non-stream VAD)
  - `python export_aed.py` (Non-stream AED, 3-class)
- Outputs are placed in `runtime/convert/out/`.

 
### Build Runtime (Host)
- `./build.sh`
  - Artifacts are copied to `out/`:
    - `libfirered_vad_stream.*`
    - `test_vad_stream`
    - `test_vad_non_stream`
    - `test_aed_non_stream`

### Build Runtime (Android)
- Build NCNN for Android first (32-bit by default):
  - `./3rd/build_android.sh 20260113 $ANDROID_NDK`
- Cross-build runtime:
  - `./build_android.sh $ANDROID_NDK`
  - Artifacts are copied to `out_android/armeabi-v7a/`

### Run C++ Executables
- Stream VAD (packed cache):
  - `./out/test_vad_stream ../convert/out/firered_vad_packed_cache_stream.ncnn.param ../convert/out/firered_vad_packed_cache_stream.ncnn.bin ../convert/out/cmvn_means_stream.bin ../convert/out/cmvn_istd_stream.bin ../../assets/hello_en.wav`
- Non-stream VAD:
  - `./out/test_vad_non_stream ../convert/out/firered_vad_non_stream.ncnn.param ../convert/out/firered_vad_non_stream.ncnn.bin ../convert/out/cmvn_means_vad.bin ../convert/out/cmvn_istd_vad.bin ../../assets/hello_en.wav`
- Non-stream AED (3-class):
  - `./out/test_aed_non_stream ../convert/out/firered_aed_non_stream.ncnn.param ../convert/out/firered_aed_non_stream.ncnn.bin ../convert/out/cmvn_means_aed.bin ../convert/out/cmvn_istd_aed.bin ../../assets/event.wav`

### Stream VAD C API
- Header: `csrc/firered_vad_stream_packed.h`
- Functions:
  - `FireredVADHandle firered_vad_create(const char* param, const char* bin, const char* cmvn_means, const char* cmvn_istd);`
  - `int firered_vad_process_stream(FireredVADHandle h, const int16_t* pcm160, int num_samples, FireredVADResult* out);`
  - `void firered_vad_reset(FireredVADHandle h);`
  - `void firered_vad_destroy(FireredVADHandle h);`
- Result struct:
  - `float confidence; bool is_speech; int frame_offset;`
- Usage notes:
  - Input is 16-bit PCM at 16kHz. Feed 10ms chunks (160 samples) repeatedly.
  - Internally uses 25ms window, 10ms shift, and a packed cache `[1,1024,19]`.
  - See `examples/test_vad_stream_packed.cpp` for a full example.

### Comparison Utilities (optional)

```shell
sh test.sh

or
- Stream VAD: `python compare_vad_stream_with_pytorch.py <wav>`
- Non-stream VAD: `python compare_vad_non_stream_with_pytorch.py <wav>`
- Non-stream AED: `python compare_aed_non_stream_with_pytorch.py <wav>`

```
