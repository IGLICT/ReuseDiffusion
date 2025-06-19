"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import functools
import operator
import torch
import triton
import triton.language as tl

aten = torch.ops.aten

@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    # w2_over_w = weight_2 / new_weight
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride: tl.constexpr,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N).to(tl.float32)
        m2_ = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        weight_ = (cols < N).to(tl.float32)
        _mean, _m2, _weight = x, m2_, weight_
    else:
        # Compute mean
        _mean = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        _m2 = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        _weight = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N).to(tl.float32)
            m2_ = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
            weight_ = (cols < N).to(tl.float32)
            if off == 0:
                _mean, _m2, _weight = x, m2_, weight_
            else:
                _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x,
                                                      m2_, weight_)
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1 / tl.sqrt(var + eps)
    mean = mean.to(x.dtype)
    rstd = rstd.to(x.dtype)
    # Write mean / rstd
    if Mean is not None:
        tl.store(Mean + row, mean)
    if Rstd is not None:
        tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if W is None:
            w = tl.full((BLOCK_SIZE, ), 1.0, dtype=x.dtype)
        else:
            w = tl.load(W + cols, mask=mask)
        if B is None:
            b = tl.zeros((BLOCK_SIZE, ), dtype=x.dtype)
        else:
            b = tl.load(B + cols, mask=mask)
        # x = tl.load(X + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)
    else:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            if W is None:
                w = tl.full((BLOCK_SIZE, ), 1.0, dtype=x.dtype)
            else:
                w = tl.load(W + cols, mask=mask)
            if B is None:
                b = tl.zeros((BLOCK_SIZE, ), dtype=x.dtype)
            else:
                b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Write output
            tl.store(Y + cols, y, mask=mask)


def triton_layernorm(x, normalized_shape, weight, bias, eps):
    x = x.contiguous()
    weight = weight.contiguous() if weight is not None else None
    bias = bias.contiguous() if bias is not None else None
    # allocate output
    y = torch.empty_like(x)
    
    N = functools.reduce(operator.mul, normalized_shape, 1)
    x_arg = x.reshape(-1, N)
    M, N = x_arg.shape
    # allocate cache
    mean = torch.empty((M, ), dtype=x.dtype, device=x.device)
    rstd = torch.empty((M, ), dtype=x.dtype, device=x.device)
    # 
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 16)
    # enqueue kernel
    _layer_norm_fwd_fused[(M,)](
        x_arg, y, weight, bias, mean, rstd,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

if __name__ == '__main__':

    def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape,
                            dtype=dtype,
                            device='cuda',
                            requires_grad=True)
        bias = torch.rand(w_shape,
                          dtype=dtype,
                          device='cuda',
                          requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
        # forward pass
        y_tri = triton_layernorm(x, w_shape, weight, bias, eps)
        y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias,
                                               eps).to(dtype)
        # compare
        assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 20)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name='layer-norm-backward',
            args={
                'M': 4096,
                'dtype': torch.float16,
                'mode': 'forward'
            },
        ))
    def bench_layer_norm(M,
                         N,
                         dtype,
                         provider,
                         mode='forward',
                         eps=1e-5,
                         device='cuda'):
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape,
                            dtype=dtype,
                            device='cuda',
                            requires_grad=True)
        bias = torch.rand(w_shape,
                          dtype=dtype,
                          device='cuda',
                          requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
        quantiles = [0.5, 0.2, 0.8]
        # utility functions
        if provider == 'triton':

            def y_fwd():
                return triton_layernorm(x, w_shape, weight, bias,
                                  eps)  # noqa: F811, E704

        if provider == 'torch':

            def y_fwd():
                return torch.nn.functional.layer_norm(x, w_shape, weight, bias,
                                                      eps)  # noqa: F811, E704

        # forward pass
        if mode == 'forward':

            def gbps(ms):
                return 2 * x.numel() * x.element_size() / ms * 1e-6

            ms, min_ms, max_ms = triton.testing.do_bench(y_fwd,
                                                         quantiles=quantiles,
                                                         rep=500)
        
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    test_layer_norm(1151, 8192, torch.float16)
    bench_layer_norm.run(save_path='.', print_data=True)

# %%
# References
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
