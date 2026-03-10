#!/usr/bin/env python3
"""
FireRedVAD NCNN 导出脚本（打包 cache 版本）

使用方法:
    python export_ncnn_packed_cache.py

输出:
    - firered_vad_packed_cache.onnx
    - firered_vad_packed_cache.ncnn.param
    - firered_vad_packed_cache.ncnn.bin

说明:
    将 8 个 cache [1,128,19] 打包成 1 个 [1,1024,19]
    解决 PNNX 多输入转换失败的问题
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fireredvad.core.detect_model import DetectModel


class FireRedVADPackedCache(nn.Module):
    """
    FireRedVAD 打包 cache 版本
    
    将 8 个 cache [1,128,19] 打包成 1 个 [1,1024,19]
    解决 PNNX 对多输入模型的转换限制
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
                          = 8 x [1, 128, 19] 打包
        Returns:
            probs: [batch=1, time=1, 1]
            new_caches_packed: [batch=1, 1024, 19]
        """
        # 拆分包 cache: [1, 1024, 19] -> 8 x [1, 128, 19]
        caches = list(caches_packed.chunk(8, dim=1))
        
        # 前向传播
        probs, new_caches = self.model.forward(feat, caches=caches)
        
        # 打包 new_caches: 8 x [1, 128, 19] -> [1, 1024, 19]
        new_caches_packed = torch.cat(new_caches, dim=1)
        
        return probs, new_caches_packed


def export_packed_cache():
    """导出打包 cache 的 ONNX 模型"""
    
    model_dir = 'pretrained_models/xukaituo/FireRedVAD/VAD'
    
    print(f"Loading model from: {model_dir}")
    
    # 创建模型
    model = FireRedVADPackedCache(model_dir)
    model.eval()
    
    # 示例输入
    feat = torch.randn(1, 1, 80)
    caches_packed = torch.randn(1, 1024, 19)  # 8 * 128 = 1024
    
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
    
    torch.onnx.export(
        model,
        (feat, caches_packed),
        'firered_vad_packed_cache.onnx',
        input_names=['feat', 'caches_packed'],
        output_names=['probs', 'new_caches_packed'],
        opset_version=11,
        do_constant_folding=True
    )
    
    print(f"Exported to: firered_vad_packed_cache.onnx")
    
    # 验证
    print("\nValidating ONNX model...")
    import onnxruntime as ort
    sess = ort.InferenceSession('firered_vad_packed_cache.onnx')
    
    print("ONNX inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    
    print("\nONNX outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape}")
    
    # 测试推理
    print("\nTesting ONNX inference...")
    feat_np = feat.numpy()
    caches_np = caches_packed.numpy()
    
    outputs = sess.run(None, {'feat': feat_np, 'caches_packed': caches_np})
    probs_np = outputs[0]
    
    print(f"ONNX output probs: {probs_np[0, 0, 0]:.4f}")
    print("\nONNX export successful!")
    
    # 转换为 NCNN
    print("\nConverting to NCNN...")
    import subprocess
    result = subprocess.run(
        ['pnnx', 'firered_vad_packed_cache.onnx'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("PNNX conversion successful!")
        
        # 检查输出文件
        import os
        bin_size = os.path.getsize('firered_vad_packed_cache.ncnn.bin')
        print(f"  .bin size: {bin_size / 1024 / 1024:.2f} MB")
        
        if bin_size > 1000000:  # > 1MB
            print("  ✓ Conversion looks good!")
        else:
            print("  ✗ Conversion may have failed (file too small)")
    else:
        print("PNNX conversion failed:")
        print(result.stderr)


if __name__ == '__main__':
    export_packed_cache()
