#!/usr/bin/env python3
"""
FireRedVAD Stream-VAD 版本 NCNN 导出脚本（打包 cache 版本）

使用方法:
    python export_ncnn_packed_cache_stream.py

输出:
    - firered_vad_packed_cache_stream.onnx
    - firered_vad_packed_cache_stream.ncnn.param
    - firered_vad_packed_cache_stream.ncnn.bin

说明:
    使用 Stream-VAD 模型权重（而非 VAD 权重）
    将 8 个 cache [1,128,19] 打包成 1 个 [1,1024,19]
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../../')
from fireredvad.core.detect_model import DetectModel


class FireRedVADPackedCacheStream(nn.Module):
    """
    FireRedVAD Stream-VAD 打包 cache 版本
    """
    
    def __init__(self, model_dir):
        super().__init__()
        self.model = DetectModel.from_pretrained(model_dir)
        self.model.eval()
        
    def forward(self, feat, caches_packed):
        """
        Args:
            feat: [batch=1, time=1, feat_dim=80]
            caches_packed: [batch=1, cache_size=1024, cache_len=19]
        Returns:
            probs: [batch=1, time=1, 1]
            new_caches_packed: [batch=1, 1024, 19]
        """
        # 拆分包 cache
        caches = list(caches_packed.chunk(8, dim=1))
        
        # 前向传播
        probs, new_caches = self.model.forward(feat, caches=caches)
        
        # 打包 new_caches
        new_caches_packed = torch.cat(new_caches, dim=1)
        
        return probs, new_caches_packed


def export_packed_cache_stream():
    """导出 Stream-VAD 打包 cache 的 ONNX 模型"""
    
    # 使用 Stream-VAD 模型目录
    model_dir = '../../pretrained_models/FireRedVAD/Stream-VAD'
    
    print(f"Loading Stream-VAD model from: {model_dir}")
    
    # 创建模型
    model = FireRedVADPackedCacheStream(model_dir)
    model.eval()
    
    # 示例输入
    feat = torch.randn(1, 1, 80)
    caches_packed = torch.randn(1, 1024, 19)
    
    # 测试前向传播
    print("Testing forward pass...")
    with torch.no_grad():
        probs, new_caches = model(feat, caches_packed)
        
    print(f"Input feat: {feat.shape}")
    print(f"Input caches_packed: {caches_packed.shape}")
    print(f"Output probs: {probs.shape}")
    print(f"Output new_caches: {new_caches.shape}")
    
    # 导出 ONNX
    print(f"\nExporting ONNX model...")
    os.makedirs('out/', exist_ok=True)
    torch.onnx.export(
        model,
        (feat, caches_packed),
        'out/firered_vad_packed_cache_stream.onnx',
        input_names=['feat', 'caches_packed'],
        output_names=['probs', 'new_caches_packed'],
        opset_version=11,
        do_constant_folding=True
    )
    
    print(f"Exported to: firered_vad_packed_cache_stream.onnx")
    
    # 转换为 NCNN
    print(f"\nConverting to NCNN with PNNX...")
    import subprocess
    subprocess.run([
        'pnnx',
        'out/firered_vad_packed_cache_stream.onnx',
        'inputshape=[1,1,80],[1,1024,19]',
        'inputname=feat,caches_packed',
        'outputname=probs,new_caches_packed'
    ])
    
    print(f"\nDone!")
    print(f"NCNN files:")
    print(f"  - out/firered_vad_packed_cache_stream.ncnn.param")
    print(f"  - out/firered_vad_packed_cache_stream.ncnn.bin")

    from fireredvad.core.audio_feat import CMVN

    cmvn = CMVN('../../pretrained_models/FireRedVAD/Stream-VAD/cmvn.ark')
    cmvn.means.astype(np.float32).tofile('./out/cmvn_means_stream.bin')
    cmvn.inverse_std_variances.astype(np.float32).tofile('./out/cmvn_istd_stream.bin')

    os.system('rm out/*.py')
    os.system('rm out/*pnnx*')



if __name__ == '__main__':
    export_packed_cache_stream()
