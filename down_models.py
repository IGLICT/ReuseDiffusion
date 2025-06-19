import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline, FluxPipeline, CogVideoXPipeline
from diffusers.utils import export_to_gif

# Model name to repository and pipeline class mapping
MODEL_MAPPING = {
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": StableDiffusionXLPipeline,
    },
    "sd3": {
        "repo": "stabilityai/stable-diffusion-3-medium-diffusers",
        "pipeline_class": StableDiffusion3Pipeline,
    },
    "flux": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "pipeline_class": FluxPipeline,
    },
    "cogvideox": {
        "repo": "THUDM/CogVideoX-5b",
        "pipeline_class": CogVideoXPipeline,
    },
}

def download_model(model_name: str, save_dir: Path) -> None:
    """Download the specified model and save it locally.
    
    Args:
        model_name: Name of the model to download
        save_dir: Directory to save the downloaded model
        
    Raises:
        ValueError: If the model name is not supported
    """
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"❌ Unsupported model name: {model_name}")

    model_info = MODEL_MAPPING[model_name]
    repo = model_info["repo"]
    pipeline_class = model_info["pipeline_class"]
    
    model_dir = save_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pipe = pipeline_class.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        variant="fp16" if model_name in ["sdxl"] else None,
    )
    pipe.save_pretrained(model_dir)
    print(f"✅ {model_name.upper()} downloaded and saved to {model_dir}")
    
def test_model(model_name: str, save_dir: Path) -> None:
    """Test the specified model with a sample generation.
    
    Args:
        model_name: Name of the model to test
        save_dir: Directory where the model is saved
    """
    model_info = MODEL_MAPPING[model_name]
    pipeline_class = model_info["pipeline_class"]
    
    model_dir = save_dir / model_name

    pipe = pipeline_class.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    ).to("cuda")
    
    output_dir = Path('./results') / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_prompt = 'A full view of a shower with glass.'
    
    if model_name in ["sdxl", 'sd3', 'flux']:
        output = pipe(
            prompt=test_prompt,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
        output.save(f'{output_dir}/test.jpg')
    elif model_name in ['cogvideox']:
        output = pipe(
            prompt=test_prompt,
            generator=torch.Generator(device="cuda").manual_seed(42),
            num_frames=9,
        ).frames[0]
        export_to_gif(output, f'{output_dir}/test.gif')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion model download and management tool")
    parser.add_argument(
        "--mode",
        choices=["download", "test"],
        default="download",
        help="Operation mode: download models or test existing ones"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["sdxl", "sd3", "flux", "cogvideox"],
        required=True,
        help="Specify models to process (supports multiple selections)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Root directory to save/download models"
    )
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    for model_name in args.models:
        try:
            if args.mode == 'download':
                download_model(model_name, save_dir)
            else:
                test_model(model_name, save_dir)
        except Exception as e:
            print(f"❌ Error processing {model_name}: {str(e)}")