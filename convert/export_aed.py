#!/usr/bin/env python3
"""
FireRedVAD AED (3-class: speech/singing/music) -> Export ONNX and NCNN (non-stream, no cache)

Model dir: ../../pretrained_models/FireRedVAD/AED

Outputs:
  - out/firered_aed_non_stream.onnx
  - out/firered_aed_non_stream.ncnn.param
  - out/firered_aed_non_stream.ncnn.bin
  - out/cmvn_means_aed.bin, out/cmvn_istd_aed.bin

Run:
  cd runtime/convert
  python export_aed.py
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')
from fireredvad.core.detect_model import DetectModel
from fireredvad.core.audio_feat import CMVN


class FireRedAEDNonStream(nn.Module):
    """Non-stream AED wrapper (no caches), output 3-class probs per frame."""

    def __init__(self, model_dir: str):
        super().__init__()
        self.model = DetectModel.from_pretrained(model_dir)
        self.model.eval()

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: [batch=1, time=T, feat_dim=80]
        Returns:
            probs: [batch=1, time=T, 3]  # (speech, singing, music)
        """
        probs, _ = self.model.forward(feat, caches=None)
        return probs


def export_aed_non_stream():
    model_dir = '../../pretrained_models/FireRedVAD/AED'
    print(f"Loading AED (non-stream) model from: {model_dir}")

    model = FireRedAEDNonStream(model_dir)
    model.eval()

    # Dummy input with time length 100
    feat = torch.randn(1, 100, 80)

    print("Testing forward pass...")
    with torch.no_grad():
        probs = model(feat)
    print(f"Input feat:  {feat.shape}")
    print(f"Output probs:{probs.shape}")

    os.makedirs('out', exist_ok=True)

    # Export ONNX
    onnx_path = 'out/firered_aed_non_stream.onnx'
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

    print("\nExporting CMVN for AED...")
    cmvn = CMVN(os.path.join(model_dir, 'cmvn.ark'))
    cmvn.means.astype('float32').tofile('./out/cmvn_means_aed.bin')
    cmvn.inverse_std_variances.astype('float32').tofile('./out/cmvn_istd_aed.bin')

    # Clean pnnx python stubs if any
    os.system('rm -f out/*.py')
    os.system('rm -f out/*pnnx*')

    print("\nDone!")
    print("NCNN files:")
    print("  - out/firered_aed_non_stream.ncnn.param")
    print("  - out/firered_aed_non_stream.ncnn.bin")


if __name__ == '__main__':
    export_aed_non_stream()
