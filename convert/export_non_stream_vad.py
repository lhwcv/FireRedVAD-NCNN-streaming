#!/usr/bin/env python3
"""
FireRedVAD 非流式 VAD -> 导出 ONNX 与 NCNN（不带缓存）

使用模型目录: ../../pretrained_models/FireRedVAD/VAD

输出:
  - out/firered_vad_non_stream.onnx
  - out/firered_vad_non_stream.ncnn.param
  - out/firered_vad_non_stream.ncnn.bin
  - out/cmvn_means_vad.bin, out/cmvn_istd_vad.bin

运行:
  cd runtime/convert
  python export_non_stream_vad.py
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')
from fireredvad.core.detect_model import DetectModel
from fireredvad.core.audio_feat import CMVN


class FireRedVADNonStream(nn.Module):
    """非流式 VAD 封装（不含 cache）。"""

    def __init__(self, model_dir: str):
        super().__init__()
        self.model = DetectModel.from_pretrained(model_dir)
        self.model.eval()

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: [batch=1, time=T, feat_dim=80]
        Returns:
            probs: [batch=1, time=T, 1]
        """
        probs, _ = self.model.forward(feat, caches=None)
        return probs


def export_non_stream():
    model_dir = '../../pretrained_models/FireRedVAD/VAD'
    print(f"Loading VAD (non-stream) model from: {model_dir}")

    model = FireRedVADNonStream(model_dir)
    model.eval()

    # Dummy input: sequence length 100
    feat = torch.randn(1, 100, 80)

    print("Testing forward pass...")
    with torch.no_grad():
        probs = model(feat)
    print(f"Input feat:  {feat.shape}")
    print(f"Output probs:{probs.shape}")

    os.makedirs('out', exist_ok=True)

    # Export ONNX (no caches)
    onnx_path = 'out/firered_vad_non_stream.onnx'
    print("\nExporting ONNX model...")
    torch.onnx.export(
        model,
        (feat,),
        onnx_path,
        input_names=['feat'],
        output_names=['probs'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={'feat': {1: 'time'}, 'probs': {1: 'time'}},
    )
    print(f"Exported to: {onnx_path}")

    # Convert to NCNN with pnnx
    print("\nConverting to NCNN with PNNX...")
    subprocess.run([
        'pnnx',
        onnx_path,
        'inputshape=[1,100,80]',
        'inputname=feat',
        'outputname=probs'
    ])

    print("\nExporting CMVN for VAD...")
    cmvn = CMVN(os.path.join(model_dir, 'cmvn.ark'))
    cmvn.means.astype('float32').tofile('./out/cmvn_means.bin')
    cmvn.inverse_std_variances.astype('float32').tofile('./out/cmvn_istd.bin')

    # Clean pnnx python stubs if any
    os.system('rm -f out/*.py')
    os.system('rm -f out/*pnnx*')

    print("\nDone!")
    print("NCNN files:")
    print("  - out/firered_vad_non_stream.ncnn.param")
    print("  - out/firered_vad_non_stream.ncnn.bin")


if __name__ == '__main__':
    export_non_stream()
