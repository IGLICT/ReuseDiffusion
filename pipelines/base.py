from abc import ABC
import json
from pathlib import Path
from typing import List, Optional
import torch

class BaseReusePipeline(ABC):
    """Base class for pipelines with reuse functionality.
    
    This class provides caching and reuse mechanisms for diffusion model pipelines,
    allowing for step skipping based on historical difference patterns.
    """
    
    def __init__(self, model: str):
        """Initialize the pipeline with caching variables.
        
        Args:
            model: Name of the model (used for history tracking)
        """
        self.model = model
        self.max_skip_steps: Optional[int] = None
        self.threshold: Optional[int] = None
        self.collect_diff: bool = False
        self.reset_cache()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, 
                       max_skip_steps: Optional[int] = None,
                       threshold: Optional[int] = None,
                       collect_diff: bool = False,
                       **kwargs):
        """Create pipeline instance with reuse configuration.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            max_skip_steps: Maximum consecutive steps that can be skipped
            threshold: Percentage threshold for historical skipping (0-100)
            collect_diff: Whether to collect difference statistics
            **kwargs: Additional arguments for parent class
            
        Returns:
            Configured pipeline instance
        """
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipeline.__class__ = cls
        pipeline.max_skip_steps = max_skip_steps
        pipeline.threshold = threshold
        pipeline.collect_diff = collect_diff
        return pipeline
    
    def reset_cache(self) -> None:
        """Reset all caching variables to their initial state."""
        self.noise_pred: Optional[torch.Tensor] = None
        self.prev_latents: List[torch.Tensor] = []
        self.diff_list: List[torch.Tensor] = []
        self.diff_l2_list: List[torch.Tensor] = []
        self.diff_l3_list: List[torch.Tensor] = []
        self.mask: List[bool] = [False]
        self.loaded_diff_l2_list: Optional[List[float]] = None
        self.loaded_threshold: Optional[float] = None
        self.sorted_list: List[float] = []
        self.load_history()
    
    def load_history(self) -> None:
        """Load historical difference data from JSON file if available."""
        if self.loaded_diff_l2_list is None:
            try:
                history_file = Path('input/history') / f"{self.model}.json"
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        data = json.load(f)
                        self.loaded_diff_l2_list = data.get('diff_l2_list', [])
                        self.sorted_list = sorted(self.loaded_diff_l2_list)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load history - {str(e)}")
                self.loaded_diff_l2_list = []
                self.sorted_list = []
    
    def estimate_skipping_from_history(self) -> bool:
        """Determine if current step can be skipped based on historical patterns.
        
        Returns:
            True if step can be skipped (diff_l2 is in bottom threshold%), False otherwise
        """
        if self.threshold is None or not self.loaded_diff_l2_list:
            return False
        
        step_index = len(self.diff_list) - 1
        if step_index < 0 or step_index >= len(self.loaded_diff_l2_list):
            return False
        
        if not self.loaded_threshold and self.sorted_list:
            index = min(int(len(self.sorted_list) * self.threshold / 100), 
                       len(self.sorted_list) - 1)
            self.loaded_threshold = self.sorted_list[index]
        
        current_value = self.loaded_diff_l2_list[step_index]
        return current_value < self.loaded_threshold if self.loaded_threshold else False
    
    def estimate_skipping(self) -> bool:
        """Determine if current step can be skipped based on recent differences.
        
        Returns:
            True if step can be skipped, False otherwise
        """
        if not self.diff_l3_list:
            return False
        
        # Prevent skipping too many consecutive steps
        if (self.max_skip_steps and len(self.mask) > self.max_skip_steps and 
            all(self.mask[-self.max_skip_steps:])):
            return False
            
        # Skip if difference is below threshold
        return self.diff_l3_list[-1] <= 0.01
    
    def update_cache(self, latent: torch.Tensor) -> None:
        """Update cache with new latent and compute differences.
        
        Args:
            latent: New latent tensor to add to cache
        """
        self.prev_latents.append(latent)
        
        # Compute first-order difference
        if len(self.prev_latents) >= 2:
            diff = (self.prev_latents[-1] - self.prev_latents[-2]).abs().mean()
            self.diff_list.append(diff)
        
        # Compute second-order difference
        if len(self.diff_list) >= 2:
            diff_l2 = (self.diff_list[-1] - self.diff_list[-2]).abs() / max(self.diff_list[-1], 1e-8)
            self.diff_l2_list.append(diff_l2)
        
        # Compute third-order difference
        if len(self.diff_list) >= 3:
            diff_l3 = ((self.diff_list[-1] + self.diff_list[-3])/2 - self.diff_list[-2]).abs() / max(self.diff_list[-2], 1e-8)
            self.diff_l3_list.append(diff_l3)
        
        # Update skip mask
        history_mask = self.estimate_skipping_from_history()
        current_mask = self.estimate_skipping()
        self.mask.append(False if self.collect_diff else (history_mask and current_mask))
    
    def __end__(self) -> None:
        """Clean up pipeline after generation completes."""
        if self.collect_diff:
            self.save_diff()
        self.reset_cache()
    
    def save_diff(self) -> None:
        """Save difference statistics to model history file.
        
        If historical data exists, averages new data with existing values.
        """
        # Prepare data for saving
        data = {
            "diff_list": [diff.item() for diff in self.diff_list],
            "diff_l2_list": [diff.item() for diff in self.diff_l2_list],
            "diff_l3_list": [diff.item() for diff in self.diff_l3_list],
        }
        
        # Create save directory
        save_dir = Path('input/history')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{self.model}.json"
        
        # Merge with existing data if available
        try:
            if save_path.exists():
                with open(save_path, 'r') as f:
                    history = json.load(f)
                    # Weighted average (could be adjusted)
                    data = {
                        key: [
                            (new + old) / 2 
                            for new, old in zip(data[key], history[key])
                        ] 
                        for key in data if key in history
                    }
        except Exception as e:
            print(f"Warning: Could not merge with history - {str(e)}")
        
        # Save data
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error: Failed to save difference data - {str(e)}")
            raise
