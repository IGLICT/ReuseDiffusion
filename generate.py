import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
from itertools import islice
import numpy as np
import random
import time
import torch

from diffusers.models import attention_processor
from attn_processor.cogvideox import ReuseCogVideoXAttnProcessor2_0
attention_processor.CogVideoXAttnProcessor2_0 = ReuseCogVideoXAttnProcessor2_0

from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline, FluxPipeline, CogVideoXPipeline

from pipelines.cogvideox import ReuseCogVideoXPipeline
from pipelines.flux import ReuseFluxPipeline
from pipelines.sdxl import ReuseStableDiffusionXLPipeline
from pipelines.sd3 import ReuseStableDiffusion3Pipeline

from input.coco import load_coco2017

from diffusers.utils import export_to_gif

MODEL_MAPPING = {
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "reuse_pipeline_class": ReuseStableDiffusionXLPipeline,
        "generation_type": "image",
    },
    "sd3": {
        "repo": "stabilityai/stable-diffusion-3-medium-diffusers",
        "pipeline_class": StableDiffusion3Pipeline,
        "reuse_pipeline_class": ReuseStableDiffusion3Pipeline,
        "generation_type": "image",
    },
    "flux": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "pipeline_class": FluxPipeline,
        "reuse_pipeline_class": ReuseFluxPipeline,
        "generation_type": "image",
    },
    "cogvideox": {
        "repo": "THUDM/CogVideoX-5b",
        "pipeline_class": CogVideoXPipeline,
        "reuse_pipeline_class": ReuseCogVideoXPipeline,
        "generation_type": "video",
    },
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def process_batch(prompts_batch, pipe, args):
    """Process a single batch of prompts and generate images/videos"""
    sample_prompts = [p["prompt"] for p in prompts_batch]
    filenames = [p["filename"] for p in prompts_batch]
    
    # Generate content
    if args.benchmark:
        start_time = time.perf_counter()
    if args.generation_type == "image":
        generated_images = pipe(sample_prompts, num_inference_steps=args.steps,
                                width=args.width_height, height=args.width_height)
        generated_data = generated_images.images
    elif args.generation_type == "video":
        generated_videos = pipe(sample_prompts, num_inference_steps=args.steps, num_frames=9,
                                width=args.width_height, height=args.width_height)
        generated_data = generated_videos.frames
    else:
        raise ValueError(f"❌ Unsupported generation type: {args.generation_type}")
    if args.benchmark:
        elapsed_time = time.perf_counter() - start_time
        print(f"⏱ Single inference time: {elapsed_time:.2f} seconds")
    return generated_data, filenames

def save_data(data, filenames, model_dir, generation_type):
    for item, filename in zip(data, filenames):
        file_path = model_dir.joinpath(filename)
        if generation_type == "image":
            file_path = f'{file_path}.jpg'
            item.save(file_path)
        elif generation_type == "video":
            file_path = f'{file_path}.gif'
            export_to_gif(item, file_path)
        else:
            raise ValueError(f"❌ Unsupported generation type: {generation_type}")


def generate_data(prompts, pipe, args, output_dir):
    """generate and save images/videos"""
    # Determine output directory
    output_subdir = args.mode
    model_dir = Path(output_dir).joinpath(args.model, output_subdir)
    
    if args.mode == 'reuse':
        model_dir = model_dir.joinpath(f'max_skip_step-{args.max_skip_steps}-threshold-{args.threshold}')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize config list (only used for reuse diffusion)
    config_list = [] if args.mode == 'reuse' else None
    
    # Process prompts in batches
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    start_time = time.perf_counter()
    
    for i in range(num_batches):
        # Get current batch of prompts
        batch_start = i * args.batch_size
        batch_end = batch_start + args.batch_size
        prompts_batch = list(islice(prompts, batch_start, batch_end))
        
        # Generate batch
        generated_data, filenames = process_batch(prompts_batch, pipe, args)
        
        # Save generated content
        save_data(generated_data, filenames, model_dir, args.generation_type)
        
    
    elapsed_time = time.perf_counter() - start_time
    print(f"✅ Generation completed. Total time: {elapsed_time:.2f} seconds")


def load_prompts(dataset_name, prompts_num):
    """Load prompts from the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load from
        prompts_num: Number of prompts to load
        
    Returns:
        List of prompt dictionaries
    """
    if dataset_name == 'caption1000':
        with open('input/caption1000.txt') as f:
            prompts = f.readlines()
            return [{'prompt': prompt.strip()} for prompt in prompts[:prompts_num]]
    elif dataset_name == 'coco2017':
        return load_coco2017()[:prompts_num]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def initialize_pipeline(model_name, model_dir, mode, threshold=None, 
                       max_skip_steps=None, collect_diff=False, gpu_id=0):
    """Initialize the appropriate pipeline based on configuration.
    
    Args:
        model_name: Name of the model to load
        model_dir: Directory containing model files
        mode: Operation mode ('original' or 'reuse')
        threshold: Reuse threshold percentage
        max_skip_steps: Maximum steps to skip during reuse
        collect_diff: Whether to collect difference statistics
        gpu_id: GPU device ID to use
        
    Returns:
        Initialized pipeline instance
    """
    model_info = MODEL_MAPPING[model_name]
    pipeline_class = model_info["reuse_pipeline_class" if mode == 'reuse' else "pipeline_class"]
    model_path = Path(model_dir) / Path(model_name)
    
    pipeline_args = {
        'pretrained_model_name_or_path': model_path,
        'torch_dtype': torch.float16
    }
    
    if mode == 'reuse':
        pipeline_args.update({
            'threshold': threshold,
            'max_skip_steps': max_skip_steps,
            'collect_diff': collect_diff
        })
    
    pipe = pipeline_class.from_pretrained(**pipeline_args).to(f"cuda:{gpu_id}")
    return pipe

def main(args):
    """Main execution function for the image generation pipeline.
    
    Args:
        args: Command line arguments containing configuration parameters
    """
    # Initialize environment
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    
    try:
        # Load input prompts
        prompts = load_prompts(args.dataset, args.prompts_num)
        
        # Initialize pipeline based on mode
        pipe = initialize_pipeline(
            model_name=args.model,
            model_dir=args.model_dir,
            mode=args.mode,
            threshold=args.threshold,
            max_skip_steps=args.max_skip_steps,
            collect_diff=args.collect_diff,
            gpu_id=args.gpu
        )
        
        # Apply model-specific optimizations
        if args.mode == 'reuse' and args.model == 'cogvideox':
            from attn_processor.ln import replace_layernorm
            target_module = pipe.transformer if hasattr(pipe, 'transformer') else pipe.unet
            replace_layernorm(target_module)
        
        # Set generation type from model info
        model_info = MODEL_MAPPING[args.model]
        args.generation_type = model_info["generation_type"]
        
        # Execute generation
        generate_data(
            prompts=prompts,
            pipe=pipe,
            args=args,
            output_dir="./results"
        )
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image/Video generation pipeline configuration")
    
    # Model configuration
    parser.add_argument("--model", 
                        choices=["sdxl", "sd3", "flux", "cogvideox"], 
                        default='sdxl',
                        help="Model to use for generation")
    
    parser.add_argument("--model_dir", 
                        type=str, 
                        default="./models",
                        help="Directory containing model files")
    
    # Dataset configuration
    parser.add_argument("--dataset", 
                        choices=["caption1000", "coco2017"], 
                        default='coco2017',
                        help="Dataset to use for prompts")
    
    # Generation parameters
    parser.add_argument("--prompts_num", 
                        type=int, 
                        default=1,
                        help="Number of prompts to generate")
    
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1,
                        help="Batch size for generation")
    
    parser.add_argument("--steps", 
                        type=int, 
                        default=28,
                        help="Number of diffusion steps")
    
    parser.add_argument("--gpu", 
                        type=int, 
                        default=0,
                        help="GPU device ID to use")
    
    # Pipeline mode
    parser.add_argument("--mode",
                        choices=["original", "reuse"],
                        default="original",
                        help="Pipeline operation mode")
    
    # parser.add_argument(
    #     "--attn",
    #     choices=["original", "reuse"],
    #     default="original"
    # )
    # Feature flags
    parser.add_argument("--collect_diff", 
                        action="store_true", 
                        help="Enable collection of model statistics")
    
    # Reuse configuration
    parser.add_argument("--max_skip_steps", 
                        type=int, 
                        default=1,
                        help="Maximum number of steps to skip during reuse")
    
    parser.add_argument("--threshold", 
                        type=int, 
                        default=30, 
                        help="Maximum percentage of features to reuse (0-100)")
    
    # Output configuration
    parser.add_argument("--width_height",  
                        type=int, 
                        default=1024, 
                        help="Resolution of generated content (assumes square output)")
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="Random seed for reproducibility")
    
    # Performance testing
    parser.add_argument("--benchmark", 
                        action="store_true", 
                        help="Enable performance benchmarking to measure inference time")
    
    args = parser.parse_args()
    main(args)