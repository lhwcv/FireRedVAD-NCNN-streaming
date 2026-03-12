#!/usr/bin/env python3
"""
Compare non-stream VAD between PyTorch (FireRedVad) and NCNN C++ (test_vad_non_stream).
"""

import sys
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add FireRedVAD project root to path
FIREREDVAD_PATH = Path('../../')
sys.path.insert(0, str(FIREREDVAD_PATH))

from fireredvad.vad import FireRedVad, FireRedVadConfig


def run_pytorch_non_stream(wav_path: str):
    model_dir = FIREREDVAD_PATH / 'pretrained_models' / 'FireRedVAD' / 'VAD'
    cfg = FireRedVadConfig(use_gpu=False)
    vad = FireRedVad.from_pretrained(str(model_dir), cfg)
    _, probs = vad.detect(wav_path, do_postprocess=False)
    return probs.cpu().numpy().astype(np.float32)


def parse_ncnn_cpp_output(output_text: str):
    probs = []
    for line in output_text.split('\n'):
        m = re.search(r'Frame\s+(\d+):\s+time=([\d.]+)s,\s+confidence=([\d.]+)', line)
        if m:
            probs.append(float(m.group(3)))
    return np.array(probs, dtype=np.float32) if probs else None


def run_ncnn_cpp_non_stream(wav_path: str):
    this_dir = Path(__file__).resolve().parent  # runtime/ncnn
    exe = this_dir / 'out' / 'test_vad_non_stream'
    param = this_dir / '..' / 'convert' / 'out' / 'firered_vad_non_stream.ncnn.param'
    binf  = this_dir / '..' / 'convert' / 'out' / 'firered_vad_non_stream.ncnn.bin'
    means = this_dir / '..' / 'convert' / 'out' / 'cmvn_means.bin'
    istd  = this_dir / '..' / 'convert' / 'out' / 'cmvn_istd.bin'

    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found. Build with ./build.sh first.")

    cmd = [str(exe), str(param), str(binf), str(means), str(istd), str(wav_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"NCNN C++ exited with {res.returncode}:\n{res.stderr}")
    return parse_ncnn_cpp_output(res.stdout)


def compare(wav_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_name = Path(wav_path).stem

    print(f"PyTorch (non-stream) ...")
    pt_probs = run_pytorch_non_stream(wav_path)
    print(f"  PyTorch: frames={len(pt_probs)} range=[{pt_probs.min():.4f}, {pt_probs.max():.4f}] mean={pt_probs.mean():.4f}")

    print(f"NCNN C++ (non-stream) ...")
    ncnn_probs = run_ncnn_cpp_non_stream(wav_path)
    if ncnn_probs is None or len(ncnn_probs) == 0:
        raise RuntimeError("Failed to parse NCNN C++ output")
    print(f"  NCNN:    frames={len(ncnn_probs)} range=[{ncnn_probs.min():.4f}, {ncnn_probs.max():.4f}] mean={ncnn_probs.mean():.4f}")

    L = min(len(pt_probs), len(ncnn_probs))
    if L == 0:
        raise RuntimeError("No overlapping frames to compare")

    diff = np.abs(pt_probs[:L] - ncnn_probs[:L])
    print(f"\nDiff (PyTorch vs NCNN C++): max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    ax1 = axes[0]
    ax1.plot(pt_probs, label='PyTorch', linewidth=1.2)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.set_title(f'PyTorch VAD (non-stream) - {wav_name}')

    ax2 = axes[1]
    ax2.plot(ncnn_probs, label='NCNN C++', color='m', linewidth=1.2)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.set_title(f'NCNN C++ VAD (non-stream) - {wav_name}')

    plt.tight_layout()
    out_png = output_dir / f'compare_vad_non_stream_{wav_name}.png'
    plt.savefig(out_png, dpi=150)
    print(f"Plot saved to: {out_png}")

    # Save summary
    out_txt = output_dir / f'compare_vad_non_stream_{wav_name}.txt'
    with open(out_txt, 'w') as f:
        f.write(f'Non-stream VAD comparison\n')
        f.write(f'Audio: {wav_path}\n')
        f.write(f'PyTorch:  frames={len(pt_probs)} range=[{pt_probs.min():.6f}, {pt_probs.max():.6f}] mean={pt_probs.mean():.6f}\n')
        f.write(f'NCNN C++: frames={len(ncnn_probs)} range=[{ncnn_probs.min():.6f}, {ncnn_probs.max():.6f}] mean={ncnn_probs.mean():.6f}\n')
        f.write(f'Diff: max={diff.max():.6f}, mean={diff.mean():.6f}\n')
    print(f"Data saved to: {out_txt}")


if __name__ == '__main__':
    default_wav = FIREREDVAD_PATH / 'assets' / 'hello_zh.wav'
    wav = sys.argv[1] if len(sys.argv) > 1 else str(default_wav)
    compare(wav, Path('./out/'))

