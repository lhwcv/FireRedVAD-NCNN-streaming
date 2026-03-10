#!/usr/bin/env python3
"""
对比 PyTorch、ONNX、NCNN Python、NCNN C++ 四者的流式 VAD 结果（不带后处理）
"""

import numpy as np
import torch
import onnxruntime as ort
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# 添加 FireRedVAD 路径
FIREREDVAD_PATH = Path('/root/deploy/3rd/FireRedVAD')
sys.path.insert(0, str(FIREREDVAD_PATH))

from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from fireredvad.core.audio_feat import AudioFeat

# 尝试导入 NCNN
try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    NCNN_AVAILABLE = False
    print("Warning: ncnn not available, skipping NCNN comparison")


def run_pytorch_streaming(feat):
    """PyTorch 流式处理（使用 cache）"""
    model_dir = FIREREDVAD_PATH / "pretrained_models" / "xukaituo" / "FireRedVAD" / "VAD"
    config = FireRedStreamVadConfig(use_gpu=False)
    vad = FireRedStreamVad.from_pretrained(str(model_dir), config)
    vad.vad_model.eval()
    
    probs_list = []
    caches = None
    
    for i in range(feat.shape[0]):
        frame = feat[i:i+1].unsqueeze(0)  # [1, 1, 80]
        
        with torch.no_grad():
            probs, caches = vad.vad_model.forward(frame, caches=caches)
            probs_list.append(probs[0, 0, 0].item())
    
    return np.array(probs_list)


def run_onnx_streaming(feat):
    """ONNX 流式处理（使用 cache）"""
    sess = ort.InferenceSession(str(FIREREDVAD_PATH / 'firered_vad_with_cache.onnx'))
    
    probs_list = []
    caches = [np.zeros((1, 128, 19), dtype=np.float32) for _ in range(8)]
    
    for i in range(feat.shape[0]):
        frame = feat[i:i+1].unsqueeze(0).numpy()  # [1, 1, 80]
        
        inputs = {'feat': frame}
        for j in range(8):
            inputs[f'cache_{j}'] = caches[j]
        
        outputs = sess.run(None, inputs)
        probs_list.append(outputs[0][0, 0, 0])
        caches = list(outputs[1:])
    
    return np.array(probs_list)


def run_ncnn_python_streaming(feat):
    """NCNN Python 流式处理（使用 packed cache）"""
    if not NCNN_AVAILABLE:
        return None
    
    net = ncnn.Net()
    ret = net.load_param(str(FIREREDVAD_PATH / 'firered_vad_packed_cache.ncnn.param'))
    if ret != 0:
        print(f"NCNN load_param failed: {ret}")
        return None
    
    ret = net.load_model(str(FIREREDVAD_PATH / 'firered_vad_packed_cache.ncnn.bin'))
    if ret != 0:
        print(f"NCNN load_model failed: {ret}")
        return None
    
    probs_list = []
    caches_packed = np.zeros((1, 1024, 19), dtype=np.float32)
    
    for i in range(feat.shape[0]):
        frame = feat[i:i+1].unsqueeze(0).numpy()  # [1, 1, 80]
        
        ex = net.create_extractor()
        ex.input('in0', ncnn.Mat(frame[0]))  # [1, 80]
        ex.input('in1', ncnn.Mat(caches_packed[0]))  # [1024, 19]
        
        ret, probs_mat = ex.extract('out0')
        if ret != 0:
            print(f"NCNN extract failed at frame {i}")
            break
        
        ret, new_cache_mat = ex.extract('out1')
        
        probs_list.append(probs_mat[0])
        caches_packed = np.array(new_cache_mat).reshape(1, 1024, 19)
    
    return np.array(probs_list)


def run_ncnn_cpp(raw_file):
    """读取 NCNN C++ 的输出"""
    probs_list = []
    
    with open(raw_file, 'r') as f:
        for line in f:
            match = re.search(r'Frame\s+(\d+):\s+time=([\d.]+)s,\s+confidence=([\d.]+)', line)
            if match:
                frame_idx = int(match.group(1))
                confidence = float(match.group(3))
                probs_list.append(confidence)
    
    return np.array(probs_list)


def compare_four(wav_file, output_dir):
    """四者对比"""
    print(f"Processing {wav_file}...")
    print("="*60)
    
    # 提取特征
    cmvn_ark = str(FIREREDVAD_PATH / "pretrained_models" / "xukaituo" / "FireRedVAD" / "VAD" / "cmvn.ark")
    audio_feat = AudioFeat(cmvn_ark)
    feat, dur = audio_feat.extract(wav_file)
    
    print(f"Audio: {feat.shape[0]} frames, {dur:.3f}s")
    print()
    
    # PyTorch
    print("Running PyTorch streaming...")
    pt_probs = run_pytorch_streaming(feat)
    print(f"  Range: [{pt_probs.min():.4f}, {pt_probs.max():.4f}], mean={pt_probs.mean():.4f}")
    
    # ONNX
    print("Running ONNX streaming...")
    onnx_probs = run_onnx_streaming(feat)
    print(f"  Range: [{onnx_probs.min():.4f}, {onnx_probs.max():.4f}], mean={onnx_probs.mean():.4f}")
    
    # NCNN Python
    print("Running NCNN Python streaming...")
    ncnn_py_probs = run_ncnn_python_streaming(feat)
    if ncnn_py_probs is not None:
        print(f"  Range: [{ncnn_py_probs.min():.4f}, {ncnn_py_probs.max():.4f}], mean={ncnn_py_probs.mean():.4f}")
    
    # NCNN C++
    print("Running NCNN C++...")
    ncnn_cpp_raw = f"{output_dir}/ncnn_cpp_raw.txt"
    ncnn_cpp_probs = run_ncnn_cpp(ncnn_cpp_raw)
    print(f"  Range: [{ncnn_cpp_probs.min():.4f}, {ncnn_cpp_probs.max():.4f}], mean={ncnn_cpp_probs.mean():.4f}")
    
    # 对齐（取最小长度）
    min_len = min(len(pt_probs), len(onnx_probs))
    if ncnn_py_probs is not None:
        min_len = min(min_len, len(ncnn_py_probs))
    min_len = min(min_len, len(ncnn_cpp_probs))
    
    pt_probs = pt_probs[:min_len]
    onnx_probs = onnx_probs[:min_len]
    if ncnn_py_probs is not None:
        ncnn_py_probs = ncnn_py_probs[:min_len]
    ncnn_cpp_probs = ncnn_cpp_probs[:min_len]
    
    times = np.arange(min_len) * 0.01
    
    print()
    print("="*60)
    print("对比结果（前 {} 帧）:".format(min_len))
    print("="*60)
    
    # PyTorch vs ONNX
    diff_pt_onnx = np.abs(pt_probs - onnx_probs)
    print(f"\nPyTorch vs ONNX:")
    print(f"  Max diff:  {diff_pt_onnx.max():.6f}")
    print(f"  Mean diff: {diff_pt_onnx.mean():.6f}")
    
    # PyTorch vs NCNN Python
    if ncnn_py_probs is not None:
        diff_pt_ncnn_py = np.abs(pt_probs - ncnn_py_probs)
        print(f"\nPyTorch vs NCNN Python:")
        print(f"  Max diff:  {diff_pt_ncnn_py.max():.6f}")
        print(f"  Mean diff: {diff_pt_ncnn_py.mean():.6f}")
    
    # PyTorch vs NCNN C++
    diff_pt_ncnn_cpp = np.abs(pt_probs - ncnn_cpp_probs)
    print(f"\nPyTorch vs NCNN C++:")
    print(f"  Max diff:  {diff_pt_ncnn_cpp.max():.6f}")
    print(f"  Mean diff: {diff_pt_ncnn_cpp.mean():.6f}")
    
    # ONNX vs NCNN C++
    diff_onnx_ncnn_cpp = np.abs(onnx_probs - ncnn_cpp_probs)
    print(f"\nONNX vs NCNN C++:")
    print(f"  Max diff:  {diff_onnx_ncnn_cpp.max():.6f}")
    print(f"  Mean diff: {diff_onnx_ncnn_cpp.mean():.6f}")
    
    # 语音帧一致性（阈值 0.5）
    print(f"\n{'='*60}")
    print("语音帧检测（threshold=0.5）:")
    print(f"{'='*60}")
    
    pt_speech = pt_probs >= 0.5
    onnx_speech = onnx_probs >= 0.5
    ncnn_cpp_speech = ncnn_cpp_probs >= 0.5
    
    print(f"PyTorch:        {np.sum(pt_speech)} frames ({np.sum(pt_speech)/len(pt_speech)*100:.1f}%)")
    print(f"ONNX:           {np.sum(onnx_speech)} frames ({np.sum(onnx_speech)/len(onnx_speech)*100:.1f}%)")
    if ncnn_py_probs is not None:
        ncnn_py_speech = ncnn_py_probs >= 0.5
        print(f"NCNN Python:    {np.sum(ncnn_py_speech)} frames ({np.sum(ncnn_py_speech)/len(ncnn_py_speech)*100:.1f}%)")
    print(f"NCNN C++:       {np.sum(ncnn_cpp_speech)} frames ({np.sum(ncnn_cpp_speech)/len(ncnn_cpp_speech)*100:.1f}%)")
    
    # 计算一致性
    agreement_pt_onnx = np.sum(pt_speech == onnx_speech) / len(pt_speech) * 100
    agreement_pt_ncnn_cpp = np.sum(pt_speech == ncnn_cpp_speech) / len(pt_speech) * 100
    agreement_onnx_ncnn_cpp = np.sum(onnx_speech == ncnn_cpp_speech) / len(onnx_speech) * 100
    
    print(f"\n语音帧一致性:")
    print(f"  PyTorch vs ONNX:      {agreement_pt_onnx:.1f}%")
    print(f"  PyTorch vs NCNN C++:  {agreement_pt_ncnn_cpp:.1f}%")
    print(f"  ONNX vs NCNN C++:     {agreement_onnx_ncnn_cpp:.1f}%")
    
    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # PyTorch
    ax1 = axes[0]
    ax1.plot(times, pt_probs, 'b-', linewidth=0.8, label='PyTorch')
    ax1.fill_between(times, 0, 1, where=pt_speech, alpha=0.2, color='green', label='Speech')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Threshold')
    ax1.set_ylabel('Confidence')
    ax1.set_title(f'PyTorch - {Path(wav_file).stem} (speech: {np.sum(pt_speech)} frames)')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # ONNX
    ax2 = axes[1]
    ax2.plot(times, onnx_probs, 'b-', linewidth=0.8, label='ONNX')
    ax2.fill_between(times, 0, 1, where=onnx_speech, alpha=0.2, color='green', label='Speech')
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Threshold')
    ax2.set_ylabel('Confidence')
    ax2.set_title(f'ONNX - {Path(wav_file).stem} (speech: {np.sum(onnx_speech)} frames)')
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # NCNN Python
    ax3 = axes[2]
    if ncnn_py_probs is not None:
        ax3.plot(times, ncnn_py_probs, 'b-', linewidth=0.8, label='NCNN Python')
        ax3.fill_between(times, 0, 1, where=ncnn_py_speech, alpha=0.2, color='green', label='Speech')
        ax3.set_ylabel('Confidence')
        ax3.set_title(f'NCNN Python - {Path(wav_file).stem} (speech: {np.sum(ncnn_py_speech)} frames)')
    else:
        ax3.text(0.5, 0.5, 'NCNN not available', ha='center', va='center', fontsize=16)
        ax3.set_title('NCNN Python - Not Available')
    ax3.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # NCNN C++
    ax4 = axes[3]
    ax4.plot(times, ncnn_cpp_probs, 'b-', linewidth=0.8, label='NCNN C++')
    ax4.fill_between(times, 0, 1, where=ncnn_cpp_speech, alpha=0.2, color='green', label='Speech')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Confidence')
    ax4.set_title(f'NCNN C++ - {Path(wav_file).stem} (speech: {np.sum(ncnn_cpp_speech)} frames)')
    ax4.legend()
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/compare_four.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    # 保存详细数据
    data_path = f"{output_dir}/compare_four.txt"
    with open(data_path, 'w') as f:
        f.write(f"FireRedVAD 四者对比\n")
        f.write(f"Audio: {wav_file}\n")
        f.write(f"Frames: {min_len}\n\n")
        
        f.write(f"PyTorch:\n")
        f.write(f"  Range: [{pt_probs.min():.6f}, {pt_probs.max():.6f}]\n")
        f.write(f"  Mean:  {pt_probs.mean():.6f}\n")
        f.write(f"  Speech frames: {np.sum(pt_speech)}\n\n")
        
        f.write(f"ONNX:\n")
        f.write(f"  Range: [{onnx_probs.min():.6f}, {onnx_probs.max():.6f}]\n")
        f.write(f"  Mean:  {onnx_probs.mean():.6f}\n")
        f.write(f"  Speech frames: {np.sum(onnx_speech)}\n\n")
        
        if ncnn_py_probs is not None:
            f.write(f"NCNN Python:\n")
            f.write(f"  Range: [{ncnn_py_probs.min():.6f}, {ncnn_py_probs.max():.6f}]\n")
            f.write(f"  Mean:  {ncnn_py_probs.mean():.6f}\n")
            f.write(f"  Speech frames: {np.sum(ncnn_py_speech)}\n\n")
        
        f.write(f"NCNN C++:\n")
        f.write(f"  Range: [{ncnn_cpp_probs.min():.6f}, {ncnn_cpp_probs.max():.6f}]\n")
        f.write(f"  Mean:  {ncnn_cpp_probs.mean():.6f}\n")
        f.write(f"  Speech frames: {np.sum(ncnn_cpp_speech)}\n\n")
        
        f.write(f"Differences:\n")
        f.write(f"  PyTorch vs ONNX:     max={diff_pt_onnx.max():.6f}, mean={diff_pt_onnx.mean():.6f}\n")
        if ncnn_py_probs is not None:
            f.write(f"  PyTorch vs NCNN Py:  max={diff_pt_ncnn_py.max():.6f}, mean={diff_pt_ncnn_py.mean():.6f}\n")
        f.write(f"  PyTorch vs NCNN C++: max={diff_pt_ncnn_cpp.max():.6f}, mean={diff_pt_ncnn_cpp.mean():.6f}\n")
        f.write(f"  ONNX vs NCNN C++:    max={diff_onnx_ncnn_cpp.max():.6f}, mean={diff_onnx_ncnn_cpp.mean():.6f}\n\n")
        
        f.write(f"Speech Frame Agreement:\n")
        f.write(f"  PyTorch vs ONNX:     {agreement_pt_onnx:.1f}%\n")
        if ncnn_py_probs is not None:
            f.write(f"  PyTorch vs NCNN Py:  {agreement_pt_onnx:.1f}%\n")
        f.write(f"  PyTorch vs NCNN C++: {agreement_pt_ncnn_cpp:.1f}%\n")
        f.write(f"  ONNX vs NCNN C++:    {agreement_onnx_ncnn_cpp:.1f}%\n")
    
    print(f"Data saved to: {data_path}")
    
    return {
        'max_diff_pt_ncnn_cpp': diff_pt_ncnn_cpp.max(),
        'mean_diff_pt_ncnn_cpp': diff_pt_ncnn_cpp.mean(),
        'agreement_pt_ncnn_cpp': agreement_pt_ncnn_cpp
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_four_streaming.py <wav_file> <output_dir>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./compare_four_output"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = compare_four(wav_file, output_dir)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"PyTorch vs NCNN C++:")
    print(f"  Max Diff:       {results['max_diff_pt_ncnn_cpp']:.6f}")
    print(f"  Mean Diff:      {results['mean_diff_pt_ncnn_cpp']:.6f}")
    print(f"  Agreement:      {results['agreement_pt_ncnn_cpp']:.1f}%")
    
    if results['max_diff_pt_ncnn_cpp'] < 0.01 and results['agreement_pt_ncnn_cpp'] > 95:
        print("\n✅ PASS: NCNN C++ matches PyTorch!")
    elif results['max_diff_pt_ncnn_cpp'] < 0.05 and results['agreement_pt_ncnn_cpp'] > 90:
        print("\n⚠️  ACCEPTABLE: Small differences detected")
    else:
        print("\n❌ WARNING: Significant differences detected")
