#!/usr/bin/env python3
"""
仅对比：PyTorch Stream-VAD 与 NCNN C++ test_vad_stream 的流式结果。
"""

import numpy as np
import torch
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# 添加 FireRedVAD 路径
FIREREDVAD_PATH = Path('../../')
sys.path.insert(0, str(FIREREDVAD_PATH))

from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from fireredvad.core.audio_feat import AudioFeat
from fireredvad.core.vad_postprocessor import VadPostprocessor


def run_pytorch_streaming(feat, wav_dur):
    """PyTorch 流式处理（Stream-VAD 模型）"""
    model_dir = FIREREDVAD_PATH / "pretrained_models" / "FireRedVAD" / "Stream-VAD"
    config = FireRedStreamVadConfig(use_gpu=False)
    vad = FireRedStreamVad.from_pretrained(str(model_dir), config)
    vad.vad_model.eval()
    
    probs_list = []
    caches = None
    
    for i in range(feat.shape[0]):
        frame = feat[i:i+1].unsqueeze(0)
        with torch.no_grad():
            probs, caches = vad.vad_model.forward(frame, caches=caches)
            probs_list.append(probs[0, 0, 0].item())
    
    raw_probs = np.array(probs_list)
    
    postprocessor = VadPostprocessor(
        smooth_window_size=5,
        prob_threshold=0.5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0
    )
    
    decisions = postprocessor.process(raw_probs.tolist())
    segments = postprocessor.decision_to_segment(decisions, wav_dur)
    
    return raw_probs, decisions, segments


## 移除 ONNX 与 NCNN Python 比较，聚焦 PyTorch vs NCNN C++


def run_ncnn_cpp(raw_file):
    """读取 NCNN C++ 的输出（从文件）"""
    probs_list = []
    
    with open(raw_file, 'r') as f:
        for line in f:
            match = re.search(r'Frame\s+(\d+):\s+time=([\d.]+)s,\s+confidence=([\d.]+)', line)
            if match:
                probs_list.append(float(match.group(3)))
    
    return np.array(probs_list)


def run_ncnn_cpp_from_output(output_text):
    """从 C++ 程序输出文本解析概率值"""
    probs_list = []
    
    for line in output_text.split('\n'):
        match = re.search(r'Frame\s+(\d+):\s+time=([\d.]+)s,\s+confidence=([\d.]+)', line)
        if match:
            probs_list.append(float(match.group(3)))
    
    return np.array(probs_list) if probs_list else None


def compare_pytorch_vs_ncnn_cpp(wav_file, output_dir="./out"):
    """仅对比 PyTorch 与 NCNN C++（Stream-VAD 版本）"""
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    wav_name = Path(wav_file).stem
    
    print(f"Processing {wav_file}...")
    print("="*60)
    
    # 提取特征（使用 Stream-VAD CMVN）
    cmvn_ark = str(FIREREDVAD_PATH / "pretrained_models" / "FireRedVAD" / "Stream-VAD" / "cmvn.ark")
    audio_feat = AudioFeat(cmvn_ark)
    feat, dur = audio_feat.extract(wav_file)
    
    print(f"Audio: {feat.shape[0]} frames, {dur:.3f}s")
    print()
    
    # PyTorch
    print("Running PyTorch Streaming (Stream-VAD)...")
    pt_raw, pt_decisions, pt_segments = run_pytorch_streaming(feat, dur)
    print(f"  Range: [{pt_raw.min():.4f}, {pt_raw.max():.4f}], mean={pt_raw.mean():.4f}")
    print(f"  Segments: {pt_segments}")
    
    # NCNN C++
    print("Running NCNN C++ (Stream-VAD)...")
    import subprocess
    
    VAD_SLIM_DIR = Path(__file__).parent  # runtime/ncnn
    # 使用由 convert 导出的 NCNN 模型，以及 ncnn/out/test_vad_stream 可执行文件
    cpp_cmd = [
        str(VAD_SLIM_DIR / "out" / "test_vad_stream"),
        str(VAD_SLIM_DIR / "../convert/out/firered_vad_packed_cache_stream.ncnn.param"),
        str(VAD_SLIM_DIR / "../convert/out/firered_vad_packed_cache_stream.ncnn.bin"),
        str(VAD_SLIM_DIR / "../convert/out/cmvn_means_stream.bin"),
        str(VAD_SLIM_DIR / "../convert/out/cmvn_istd_stream.bin"),
        wav_file
    ]
    
    cpp_result = subprocess.run(cpp_cmd, capture_output=True, text=True)
    ncnn_cpp_raw = run_ncnn_cpp_from_output(cpp_result.stdout)
    
    if ncnn_cpp_raw is not None and len(ncnn_cpp_raw) > 0:
        print(f"  Range: [{ncnn_cpp_raw.min():.4f}, {ncnn_cpp_raw.max():.4f}], mean={ncnn_cpp_raw.mean():.4f}")
        cpp_postprocessor = VadPostprocessor(
            smooth_window_size=5,
            prob_threshold=0.5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0
        )
        ncnn_cpp_decisions = cpp_postprocessor.process(ncnn_cpp_raw.tolist())
        ncnn_cpp_segments = cpp_postprocessor.decision_to_segment(ncnn_cpp_decisions, dur)
        print(f"  Segments: {ncnn_cpp_segments}")
    else:
        print("  Failed to run NCNN C++")
        ncnn_cpp_raw = None
        ncnn_cpp_segments = None
    
    print(f"\n{'='*60}")
    print("对比结果:")
    print(f"{'='*60}")
    
    # PyTorch vs NCNN C++
    if ncnn_cpp_raw is not None:
        # 对齐长度
        min_len = min(len(pt_raw), len(ncnn_cpp_raw))
        diff_pt_ncnn_cpp = np.abs(pt_raw[:min_len] - ncnn_cpp_raw[:min_len])
        print(f"\nPyTorch vs NCNN C++:")
        print(f"  Max diff:  {diff_pt_ncnn_cpp.max():.6f}")
        print(f"  Mean diff: {diff_pt_ncnn_cpp.mean():.6f}")
        
        # 语音帧一致性
        pt_speech = pt_raw[:min_len] > 0.5
        cpp_speech = ncnn_cpp_raw[:min_len] > 0.5
        consistent = (pt_speech == cpp_speech).sum()
        print(f"  Speech frame consistency: {consistent}/{min_len} = {100*consistent/min_len:.1f}%")
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # PyTorch
    ax1 = axes[0]
    ax1.plot(pt_raw, 'b-', linewidth=0.8, label='PyTorch')
    ax1.fill_between(range(len(pt_raw)), 0, 1, where=pt_decisions, alpha=0.3, color='green')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax1.set_ylabel('Confidence')
    ax1.set_title(f'PyTorch Stream-VAD - {wav_name}\nSegments: {pt_segments}')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # NCNN C++
    ax2 = axes[1]
    if ncnn_cpp_raw is not None:
        ax2.plot(ncnn_cpp_raw, 'b-', linewidth=0.8, label='NCNN C++')
        ax2.fill_between(range(len(ncnn_cpp_raw)), 0, 1, where=ncnn_cpp_decisions, alpha=0.3, color='green')
        ax2.set_ylabel('Confidence')
        ax2.set_title(f'NCNN C++ Stream-VAD - {wav_name}\nSegments: {ncnn_cpp_segments}')
    else:
        ax2.text(0.5, 0.5, 'NCNN C++ not available', ha='center', va='center', fontsize=16)
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 四者对比
    ax3 = axes[2]
    ax3.plot(pt_raw, 'b-', linewidth=1.5, label='PyTorch', alpha=0.8)
    if ncnn_cpp_raw is not None:
        ax3.plot(ncnn_cpp_raw, 'm-', linewidth=1.5, label='NCNN C++', alpha=0.8)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Stream-VAD PyTorch vs NCNN C++')
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/compare_pt_vs_ncnncpp_{wav_name}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    # 保存数据
    data_path = f"{output_dir}/compare_pt_vs_ncnncpp_{wav_name}.txt"
    with open(data_path, 'w') as f:
        f.write(f"FireRedVAD Stream-VAD 四者对比\n")
        f.write(f"Audio: {wav_file}\n")
        f.write(f"Frames: {len(pt_raw)}\n")
        f.write(f"Duration: {dur:.3f}s\n\n")
        f.write(f"PyTorch:\n")
        f.write(f"  Range: [{pt_raw.min():.6f}, {pt_raw.max():.6f}]\n")
        f.write(f"  Mean:  {pt_raw.mean():.6f}\n")
        f.write(f"  Segments: {pt_segments}\n\n")
        
        if ncnn_cpp_raw is not None:
            f.write(f"NCNN C++:\n")
            f.write(f"  Range: [{ncnn_cpp_raw.min():.6f}, {ncnn_cpp_raw.max():.6f}]\n")
            f.write(f"  Mean:  {ncnn_cpp_raw.mean():.6f}\n")
            f.write(f"  Segments: {ncnn_cpp_segments}\n\n")
        
        f.write(f"Differences:\n")
        if ncnn_cpp_raw is not None:
            min_len = min(len(pt_raw), len(ncnn_cpp_raw))
            diff_pt_ncnn_cpp = np.abs(pt_raw[:min_len] - ncnn_cpp_raw[:min_len])
            f.write(f"  PyTorch vs NCNN C++: max={diff_pt_ncnn_cpp.max():.6f}, mean={diff_pt_ncnn_cpp.mean():.6f}\n")
    
    print(f"Data saved to: {data_path}")
    
    # return {
    #     'max_diff_pt_onnx': diff_pt_onnx.max(),
    #     'mean_diff_pt_onnx': diff_pt_onnx.mean()
    # }


if __name__ == '__main__':
    wav_file = str(FIREREDVAD_PATH / "assets" / "hello_zh.wav")
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    
    compare_pytorch_vs_ncnn_cpp(wav_file)
