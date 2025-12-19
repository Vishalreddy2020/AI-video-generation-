"""
Image Editing Service
Edits images based on text prompts using Stable Diffusion Image-to-Image.
The AI model automatically determines what to edit based on the prompt.
"""
import os
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np

class ImageEditingService:
    """
    Service for editing images based on text prompts.
    Uses img2img pipeline - AI automatically determines what to edit.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._detect_device()
        self.pipe = None
        self.model_loaded = False
        
        print(f"Image Editing Service initialized on device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return "xpu"
        except:
            pass
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load the image-to-image editing model."""
        if self.model_loaded:
            return True
        
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            
            print("Loading image-to-image editing model...")
            print("This may take a few minutes on first run (downloading ~4GB model)...")
            
            use_fp16 = self.device in ["cuda", "xpu", "mps"]
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                variant="fp16" if use_fp16 else None,
            )
            
            # Move to device
            if self.device == "cuda":
                self.pipe = self.pipe.to("cuda")
            elif self.device == "xpu":
                try:
                    import intel_extension_for_pytorch as ipex
                    self.pipe = self.pipe.to("xpu")
                except:
                    self.pipe = self.pipe.to("cpu")
                    self.pipe.enable_model_cpu_offload()
            elif self.device == "mps":
                self.pipe = self.pipe.to("mps")
            else:
                self.pipe = self.pipe.to("cpu")
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing()
            
            self.model_loaded = True
            print("✓ Image editing model loaded successfully!")
            return True
            
        except ImportError:
            print("diffusers library not installed. Install with: pip install diffusers transformers accelerate")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.75,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Edit an image based on a text prompt.
        The AI model automatically determines what parts of the image to edit based on the prompt.
        
        Args:
            image: PIL Image to edit
            prompt: Description of desired edits (e.g., "change the sky to sunset", "make it look like a painting")
            strength: How strong the edit should be (0.0-1.0, default: 0.75)
                      Lower values (0.3-0.5) = subtle changes, preserve more of original
                      Higher values (0.7-0.9) = more dramatic changes
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale (default: 7.5)
        
        Returns:
            PIL.Image: Edited image
        """
        # Load model if not loaded
        if not self.model_loaded:
            if not self._load_model():
                raise RuntimeError("Could not load image editing model")
        
        # Ensure image is RGB
        image = image.convert("RGB")
        
        # Ensure strength is in valid range
        strength = max(0.0, min(1.0, strength))
        
        print(f"Editing image with prompt: '{prompt}'")
        print(f"Strength: {strength} (AI will automatically determine what to edit)")
        if self.device == "cpu":
            print("Using CPU mode - this may take 1-3 minutes...")
            num_inference_steps = min(num_inference_steps, 20)  # Fewer steps on CPU
        
        # Generate edited image using img2img
        # The model will intelligently interpret the prompt and apply changes
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        
        edited_image = result.images[0]
        return edited_image

