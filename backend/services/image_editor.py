"""
AI Image Editor Service
Edits images based on text prompts using inpainting/editing models.
Works on any laptop (CPU/GPU auto-detection).
"""
import os
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np
import uuid

class ImageEditor:
    """
    Edits images based on text prompts using AI models.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._detect_device()
        self.pipe = None
        self.model_loaded = False
        
        print(f"Image Editor initialized on device: {self.device}")
    
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
    
    def load_model(self):
        """Load the image editing model."""
        if self.model_loaded:
            return True
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            print("Loading image editing model...")
            print("This may take a few minutes on first run (downloading ~4GB model)...")
            
            use_fp16 = self.device in ["cuda", "xpu", "mps"]
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
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
    
    def edit(
        self,
        source_image: Path,
        edit_prompt: str,
        strength: float = 0.75,
        mask: Optional[Path] = None
    ) -> Path:
        """
        Edit an image based on a text prompt.
        
        Args:
            source_image: Path to source image file
            edit_prompt: Description of desired edits
            strength: How strong the edit should be (0.0-1.0)
            mask: Optional mask image (white areas will be edited, black areas preserved)
        
        Returns:
            Path to edited image file
        """
        # Load model if not loaded
        if not self.model_loaded:
            if not self.load_model():
                raise RuntimeError("Could not load image editing model")
        
        # Load source image
        image = Image.open(source_image).convert("RGB")
        width, height = image.size
        
        # Create mask if not provided (edit entire image)
        if mask is None:
            # Create a mask that covers the entire image
            mask_image = Image.new("L", (width, height), 255)  # White = edit area
        else:
            mask_image = Image.open(mask).convert("L")
            mask_image = mask_image.resize((width, height))
        
        # Ensure strength is in valid range
        strength = max(0.0, min(1.0, strength))
        
        print(f"Editing image with prompt: {edit_prompt[:50]}...")
        if self.device == "cpu":
            print("Using CPU mode - this may take 1-3 minutes...")
        
        # Generate edited image
        with torch.no_grad():
            result = self.pipe(
                prompt=edit_prompt,
                image=image,
                mask_image=mask_image,
                strength=strength,
                num_inference_steps=20 if self.device == "cpu" else 30,
                guidance_scale=7.5,
            )
        
        edited_image = result.images[0]
        
        # Save edited image
        filename = f"edited_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / filename
        edited_image.save(output_path, "PNG")
        
        print(f"✓ Edited image saved: {output_path}")
        return output_path

