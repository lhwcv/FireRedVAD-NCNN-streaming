#!/usr/bin/env python3
"""
Compare non-stream AED (3 classes) between PyTorch and NCNN C++.
"""

import sys
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIREREDVAD_PATH = Path('../../')
sys.path.insert(0, str(FIREREDVAD_PATH))

from fireredvad.aed import FireRedAed, FireRedAedConfig


def run_pytorch_aed(wav_path: str) -> np.ndarray:
    model_dir = FIREREDVAD_PATH / 'pretrained_models' / 'FireRedVAD' / 'AED'
    cfg = FireRedAedConfig(use_gpu=False)
    aed = FireRedAed.from_pretrained(str(model_dir), cfg)
    _, probs = aed.detect(wav_path)
    return probs.cpu().numpy().astype(np.float32)  # (T,3)


def parse_cpp_output(text: str) -> np.ndarray:
    vals = []
    for line in text.split('\n'):
        m = re.search(r'speech=([\d\.]+),\s+singing=([\d\.]+),\s+music=([\d\.]+)', line)
        if m:
            vals.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return np.array(vals, dtype=np.float32) if vals else None


def run_ncnn_cpp(wav_path: str) -> np.ndarray:
    this_dir = Path(__file__).resolve().parent
    exe   = this_dir / 'out' / 'test_aed_non_stream'
    param = this_dir / '..' / 'convert' / 'out' / 'firered_aed_non_stream.ncnn.param'
    binf  = this_dir / '..' / 'convert' / 'out' / 'firered_aed_non_stream.ncnn.bin'
    means = this_dir / '..' / 'convert' / 'out' / 'cmvn_means_aed.bin'
    istd  = this_dir / '..' / 'convert' / 'out' / 'cmvn_istd_aed.bin'
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found. Build with ./build.sh first.")
    cmd = [str(exe), str(param), str(binf), str(means), str(istd), str(wav_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"NCNN C++ exit {res.returncode}:\n{res.stderr}")
    return parse_cpp_output(res.stdout)


def compare(wav: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(wav).stem

    print('PyTorch AED ...')
    pt = run_pytorch_aed(wav)  # (T,3)
    print(f'  frames={len(pt)} range=[{pt.min():.4f},{pt.max():.4f}] mean={pt.mean():.4f}')

    print('NCNN C++ AED ...')
    cpp = run_ncnn_cpp(wav)  # (T,3)
    if cpp is None or len(cpp) == 0:
        raise RuntimeError('Failed to parse C++ output')
    print(f'  frames={len(cpp)} range=[{cpp.min():.4f},{cpp.max():.4f}] mean={cpp.mean():.4f}')

    L = min(len(pt), len(cpp))
    pt, cpp = pt[:L], cpp[:L]
    diff = np.abs(pt - cpp)
    print(f'\nDiff per class (max / mean):')
    classes = ['speech','singing','music']
    for i, c in enumerate(classes):
        print(f'  {c:7s}: max={diff[:,i].max():.6f}  mean={diff[:,i].mean():.6f}')

    # Plot overlay
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    for i, c in enumerate(classes):
        ax = axes[i]
        ax.plot(pt[:,i], label=f'PyTorch-{c}')
        ax.plot(cpp[:,i], label=f'NCNN-{c}', alpha=0.8)
        ax.set_ylim(0,1)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f'AED {c} - {name}')
    plt.tight_layout()
    png = out_dir / f'compare_aed_non_stream_{name}.png'
    plt.savefig(png, dpi=150)
    print(f'Plot saved: {png}')

    # Save summary
    txt = out_dir / f'compare_aed_non_stream_{name}.txt'
    with open(txt, 'w') as f:
        f.write('AED non-stream comparison (PyTorch vs NCNN)\n')
        for i, c in enumerate(classes):
            f.write(f'{c}: max={diff[:,i].max():.6f}, mean={diff[:,i].mean():.6f}\n')
    print(f'Data saved: {txt}')


if __name__ == '__main__':
    default_wav = FIREREDVAD_PATH / 'assets' / 'event.wav'
    wav = sys.argv[1] if len(sys.argv) > 1 else str(default_wav)
    compare(wav, Path('./out'))

