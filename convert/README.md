# Convert ONNX and ncnn models for runtime

Quick download converted models from: https://github.com/lhwcv/FireRedVAD-NCNN-streaming/tree/main/models


## Convert yourself

Download models:
```bash

cd ~/FireRedVAD # FireRedVAD repo root dir
 
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
```
Dependencies:
```bash
# requires-python = ">=3.10"
# dependencies = [
#     "fireredvad",
#     "torch>=2.0.0",
#     "onnx>=1.14.0",
#     "onnxsim>=0.4.0",
#     "onnxruntime",
#     "huggingface_hub",
#     "pnnx"
# ]
```


Then run the conversion scripts:
```bash
cd ./runtime/convert/
python export_packed_cache_stream_vad.py
python export_non_stream_vad.py
python export_aed.py
```