import torch

def compile_sdfast(model, reuse_layernorm = False):
    from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)
    config = CompilationConfig.Default()
    
    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')

    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')

    config.enable_cuda_graph = True
    
    config.reuse_layernorm = reuse_layernorm
    model = compile(model, config)
    return model
    

def compile_torch(pipe):
    if hasattr(pipe, 'transformer'):
        pipe.transformer = torch.compile(pipe.transformer, mode='reduce-overhead', fullgraph=True)
    if hasattr(pipe, 'unet'):
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead', fullgraph=True)
    return pipe