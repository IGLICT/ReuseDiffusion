import torch
import torch.nn as nn
from typing import Optional, Tuple
import time
import functools
import operator

from .layernorm import triton_layernorm

class CustomLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        use_triton: bool = True
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            assert x.is_cuda, "Input must be a CUDA tensor"
            weight = self.weight if self.elementwise_affine else None
            bias = self.bias if self.elementwise_affine else None
            output = triton_layernorm(x, self.normalized_shape, weight, bias, self.eps)
        else:
            output = super().forward(x)
        
        return output

def replace_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            # 创建自定义 LayerNorm 并复制原始参数
            if child.elementwise_affine:
                device = child.weight.device
                dtype = child.weight.dtype
            else:
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
            
            new_layer = CustomLayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                dtype=dtype
            ).to(device)
            new_layer.load_state_dict(child.state_dict())
            setattr(module, name, new_layer)
        else:
            replace_layernorm(child)
            

def compare_layernorm():
    # 测试配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    shape = (2, 12514, 48, 64) # cogvideox
    normalized_shape = (shape[2], shape[3])
    eps = 1e-5
    
    # 创建输入和层
    x = torch.randn(shape, device=device)
    native_ln = nn.LayerNorm(normalized_shape, eps=eps).to(device)
    custom_ln = CustomLayerNorm(normalized_shape, eps=eps, use_triton=False).to(device)
    triton_ln = CustomLayerNorm(normalized_shape, eps=eps, use_triton=True).to(device)
    
    # 复制参数确保公平比较
    triton_ln.load_state_dict(native_ln.state_dict())
    custom_ln.load_state_dict(native_ln.state_dict())
    
    # 预热（避免首次运行开销）
    for _ in range(3):
        _ = native_ln(x)
        _ = triton_ln(x)
    
    # 数值比较
    with torch.no_grad():
        native_out = native_ln(x)
        triton_out = triton_ln(x)
        max_diff = (native_out - triton_out).abs().max().item()
        print(f"Max numerical difference: {max_diff:.6f}")
    
    # 性能测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = native_ln(x)
    torch.cuda.synchronize()
    native_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = triton_ln(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"Native PyTorch time: {native_time:.4f}s")
    print(f"Triton time: {triton_time:.4f}s")
    print(f"Speedup: {native_time / triton_time:.2f}x")

if __name__ == "__main__":
    compare_layernorm()