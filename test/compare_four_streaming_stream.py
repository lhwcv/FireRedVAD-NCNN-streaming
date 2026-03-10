#!/usr/bin/env python3
"""
对比 PyTorch、ONNX、NCNN Python、NCNN C++ 四者的流式 VAD 结果（Stream-VAD 版本）
使用官方 Stream-VAD 模型权重
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
FIREREDVAD_PATH = Path('/root/.copaw/czur_doc_intelligence/deploy/3rd/FireRedVAD')
sys.path.insert(0, str(FIREREDVAD_PATH))

from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
from fireredvad.core.audio_feat import AudioFeat
from fireredvad.core.vad_postprocessor import VadPostprocessor


def run_pytorch_streaming(feat, wav_dur):
    """PyTorch 流式处理（Stream-VAD 模型）"""
    model_dir = FIREREDVAD_PATH / "pretrained_models" / "xukaituo" / "FireRedVAD" / "Stream-VAD"
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


def run_onnx_streaming(feat, wav_dur):
    """ONNX 流式处理（Stream-VAD 版本）"""
    sess = ort.InferenceSession(str(FIREREDVAD_PATH / 'firered_vad_packed_cache_stream.onnx'))
    
    probs_list = []
    caches_packed = np.zeros((1, 1024, 19), dtype=np.float32)
    
    for i in range(feat.shape[0]):
        frame = feat[i:i+1].unsqueeze(0).numpy()
        inputs = {'feat': frame, 'caches_packed': caches_packed}
        
        outputs = sess.run(None, inputs)
        probs_list.append(outputs[0][0, 0, 0])
        caches_packed = outputs[1]
    
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


def run_ncnn_python_streaming(feat, wav_dur):
    """NCNN Python 流式处理（Stream-VAD 版本）"""
    try:
        import ncnn
    except ImportError:
        print("Warning: ncnn not available")
        return None, None, None
    
    net = ncnn.Net()
    ret = net.load_param(str(FIREREDVAD_PATH / 'firered_vad_packed_cache_stream.ncnn.param'))
    if ret != 0:
        print(f"NCNN load_param failed: {ret}")
        return None, None, None
    
    ret = net.load_model(str(FIREREDVAD_PATH / 'firered_vad_packed_cache_stream.ncnn.bin'))
    if ret != 0:
        print(f"NCNN load_model failed: {ret}")
        return None, None, None
    
    probs_list = []
    caches_packed = np.zeros((1, 1024, 19), dtype=np.float32)
    
    for i in range(feat.shape[0]):
        frame = feat[i:i+1].unsqueeze(0).numpy()
        
        ex = net.create_extractor()
        ex.input('in0', ncnn.Mat(frame[0]))
        ex.input('in1', ncnn.Mat(caches_packed[0]))
        
        ret, probs_mat = ex.extract('out0')
        if ret != 0:
            print(f"NCNN extract failed at frame {i}")
            break
        
        ret, new_cache_mat = ex.extract('out1')
        
        probs_list.append(probs_mat[0])
        caches_packed = np.array(new_cache_mat).reshape(1, 1024, 19)
    
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


def compare_four_stream(wav_file, output_dir="./output_stream"):
    """四者对比（Stream-VAD 版本）"""
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    wav_name = Path(wav_file).stem
    
    print(f"Processing {wav_file}...")
    print("="*60)
    
    # 提取特征（使用 Stream-VAD CMVN）
    cmvn_ark = str(FIREREDVAD_PATH / "pretrained_models" / "xukaituo" / "FireRedVAD" / "Stream-VAD" / "cmvn.ark")
    audio_feat = AudioFeat(cmvn_ark)
    feat, dur = audio_feat.extract(wav_file)
    
    print(f"Audio: {feat.shape[0]} frames, {dur:.3f}s")
    print()
    
    # PyTorch
    print("Running PyTorch Streaming (Stream-VAD)...")
    pt_raw, pt_decisions, pt_segments = run_pytorch_streaming(feat, dur)
    print(f"  Range: [{pt_raw.min():.4f}, {pt_raw.max():.4f}], mean={pt_raw.mean():.4f}")
    print(f"  Segments: {pt_segments}")
    
    # ONNX
    print("Running ONNX Streaming (Stream-VAD)...")
    onnx_raw, onnx_decisions, onnx_segments = run_onnx_streaming(feat, dur)
    print(f"  Range: [{onnx_raw.min():.4f}, {onnx_raw.max():.4f}], mean={onnx_raw.mean():.4f}")
    print(f"  Segments: {onnx_segments}")
    
    # NCNN Python
    print("Running NCNN Python Streaming (Stream-VAD)...")
    ncnn_py_raw, ncnn_py_decisions, ncnn_py_segments = run_ncnn_python_streaming(feat, dur)
    if ncnn_py_raw is not None:
        print(f"  Range: [{ncnn_py_raw.min():.4f}, {ncnn_py_raw.max():.4f}], mean={ncnn_py_raw.mean():.4f}")
        print(f"  Segments: {ncnn_py_segments}")
    
    # NCNN C++
    print("Running NCNN C++ (Stream-VAD)...")
    import subprocess
    
    VAD_SLIM_DIR = Path(__file__).parent.parent
    cpp_cmd = [
        str(VAD_SLIM_DIR / "build" / "test_vad_stream"),
        str(VAD_SLIM_DIR / "models" / "firered_vad_packed_cache_stream.ncnn.param"),
        str(VAD_SLIM_DIR / "models" / "firered_vad_packed_cache_stream.ncnn.bin"),
        str(VAD_SLIM_DIR / "models" / "cmvn_means.bin"),
        str(VAD_SLIM_DIR / "models" / "cmvn_istd.bin"),
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
    
    # PyTorch vs ONNX
    diff_pt_onnx = np.abs(pt_raw - onnx_raw)
    print(f"\nPyTorch vs ONNX:")
    print(f"  Max diff:  {diff_pt_onnx.max():.6f}")
    print(f"  Mean diff: {diff_pt_onnx.mean():.6f}")
    
    # PyTorch vs NCNN Python
    if ncnn_py_raw is not None:
        diff_pt_ncnn_py = np.abs(pt_raw - ncnn_py_raw)
        print(f"\nPyTorch vs NCNN Python:")
        print(f"  Max diff:  {diff_pt_ncnn_py.max():.6f}")
        print(f"  Mean diff: {diff_pt_ncnn_py.mean():.6f}")
    
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
    fig, axes = plt.subplots(5, 1, figsize=(16, 14))
    
    # PyTorch
    ax1 = axes[0]
    ax1.plot(pt_raw, 'b-', linewidth=0.8, label='PyTorch')
    ax1.fill_between(range(len(pt_raw)), 0, 1, where=pt_decisions, alpha=0.3, color='green')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax1.set_ylabel('Confidence')
    ax1.set_title(f'PyTorch Stream-VAD - {wav_name}\nSegments: {pt_segments}')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # ONNX
    ax2 = axes[1]
    ax2.plot(onnx_raw, 'b-', linewidth=0.8, label='ONNX')
    ax2.fill_between(range(len(onnx_raw)), 0, 1, where=onnx_decisions, alpha=0.3, color='green')
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax2.set_ylabel('Confidence')
    ax2.set_title(f'ONNX Stream-VAD - {wav_name}\nSegments: {onnx_segments}')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # NCNN Python
    ax3 = axes[2]
    if ncnn_py_raw is not None:
        ax3.plot(ncnn_py_raw, 'b-', linewidth=0.8, label='NCNN Python')
        ax3.fill_between(range(len(ncnn_py_raw)), 0, 1, where=ncnn_py_decisions, alpha=0.3, color='green')
        ax3.set_ylabel('Confidence')
        ax3.set_title(f'NCNN Python Stream-VAD - {wav_name}\nSegments: {ncnn_py_segments}')
    else:
        ax3.text(0.5, 0.5, 'NCNN Python not available', ha='center', va='center', fontsize=16)
    ax3.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # NCNN C++
    ax4 = axes[3]
    if ncnn_cpp_raw is not None:
        ax4.plot(ncnn_cpp_raw, 'b-', linewidth=0.8, label='NCNN C++')
        ax4.fill_between(range(len(ncnn_cpp_raw)), 0, 1, where=ncnn_cpp_decisions, alpha=0.3, color='green')
        ax4.set_ylabel('Confidence')
        ax4.set_title(f'NCNN C++ Stream-VAD - {wav_name}\nSegments: {ncnn_cpp_segments}')
    else:
        ax4.text(0.5, 0.5, 'NCNN C++ not available', ha='center', va='center', fontsize=16)
    ax4.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # 四者对比
    ax5 = axes[4]
    ax5.plot(pt_raw, 'b-', linewidth=1.5, label='PyTorch', alpha=0.8)
    ax5.plot(onnx_raw, 'g-', linewidth=1.5, label='ONNX', alpha=0.8)
    if ncnn_py_raw is not None:
        ax5.plot(ncnn_py_raw, 'r-', linewidth=1.5, label='NCNN Python', alpha=0.8)
    if ncnn_cpp_raw is not None:
        ax5.plot(ncnn_cpp_raw, 'm-', linewidth=1.5, label='NCNN C++', alpha=0.8)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Confidence')
    ax5.set_title(f'Stream-VAD Four-way Comparison')
    ax5.legend()
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/compare_four_stream_{wav_name}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    # 保存数据
    data_path = f"{output_dir}/compare_four_stream_{wav_name}.txt"
    with open(data_path, 'w') as f:
        f.write(f"FireRedVAD Stream-VAD 四者对比\n")
        f.write(f"Audio: {wav_file}\n")
        f.write(f"Frames: {len(pt_raw)}\n")
        f.write(f"Duration: {dur:.3f}s\n\n")
        
        f.write(f"PyTorch:\n")
        f.write(f"  Range: [{pt_raw.min():.6f}, {pt_raw.max():.6f}]\n")
        f.write(f"  Mean:  {pt_raw.mean():.6f}\n")
        f.write(f"  Segments: {pt_segments}\n\n")
        
        f.write(f"ONNX:\n")
        f.write(f"  Range: [{onnx_raw.min():.6f}, {onnx_raw.max():.6f}]\n")
        f.write(f"  Mean:  {onnx_raw.mean():.6f}\n")
        f.write(f"  Segments: {onnx_segments}\n\n")
        
        if ncnn_py_raw is not None:
            f.write(f"NCNN Python:\n")
            f.write(f"  Range: [{ncnn_py_raw.min():.6f}, {ncnn_py_raw.max():.6f}]\n")
            f.write(f"  Mean:  {ncnn_py_raw.mean():.6f}\n")
            f.write(f"  Segments: {ncnn_py_segments}\n\n")
        
        if ncnn_cpp_raw is not None:
            f.write(f"NCNN C++:\n")
            f.write(f"  Range: [{ncnn_cpp_raw.min():.6f}, {ncnn_cpp_raw.max():.6f}]\n")
            f.write(f"  Mean:  {ncnn_cpp_raw.mean():.6f}\n")
            f.write(f"  Segments: {ncnn_cpp_segments}\n\n")
        
        f.write(f"Differences:\n")
        f.write(f"  PyTorch vs ONNX:     max={diff_pt_onnx.max():.6f}, mean={diff_pt_onnx.mean():.6f}\n")
        if ncnn_py_raw is not None:
            f.write(f"  PyTorch vs NCNN Py:  max={diff_pt_ncnn_py.max():.6f}, mean={diff_pt_ncnn_py.mean():.6f}\n")
        if ncnn_cpp_raw is not None:
            min_len = min(len(pt_raw), len(ncnn_cpp_raw))
            diff_pt_ncnn_cpp = np.abs(pt_raw[:min_len] - ncnn_cpp_raw[:min_len])
            f.write(f"  PyTorch vs NCNN C++: max={diff_pt_ncnn_cpp.max():.6f}, mean={diff_pt_ncnn_cpp.mean():.6f}\n")
    
    print(f"Data saved to: {data_path}")
    
    return {
        'max_diff_pt_onnx': diff_pt_onnx.max(),
        'mean_diff_pt_onnx': diff_pt_onnx.mean()
    }


if __name__ == '__main__':
    wav_file = "/root/.copaw/czur_doc_intelligence/deploy/3rd/FireRedVAD/assets/hello_zh.wav"
    
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    
    results = compare_four_stream(wav_file)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"PyTorch vs ONNX:")
    print(f"  Max Diff:  {results['max_diff_pt_onnx']:.6f}")
    print(f"  Mean Diff: {results['mean_diff_pt_onnx']:.6f}")
    
    if results['max_diff_pt_onnx'] < 0.01:
        print("\n✅ PASS: ONNX matches PyTorch (Stream-VAD)!")
    else:
        print("\n⚠️  Check differences above")
