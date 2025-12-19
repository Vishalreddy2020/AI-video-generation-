"""
AI Image Generator Service
Generates images from text prompts using Stable Diffusion or similar models.
Works on any laptop (CPU/GPU auto-detection).
"""
import os
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
import uuid

class ImageGenerator:
    """
    Generates images from text prompts using AI models.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._detect_device()
        self.pipe = None
        self.model_loaded = False
        
        print(f"Image Generator initialized on device: {self.device}")
    
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
        """Load the image generation model."""
        if self.model_loaded:
            return True
        
        try:
            from diffusers import StableDiffusionPipeline
            
            print("Loading Stable Diffusion model...")
            print("This may take a few minutes on first run (downloading ~4GB model)...")
            
            use_fp16 = self.device in ["cuda", "xpu", "mps"]
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
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
            print("✓ Image generation model loaded successfully!")
            return True
            
        except ImportError:
            print("diffusers library not installed. Install with: pip install diffusers transformers accelerate")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        size: str = "512x512",
        seed: Optional[int] = None,
        style: Optional[str] = None,
        overlay_text: Optional[str] = None
    ) -> Path:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            size: Image size (e.g., "512x512", "768x768", "1024x1024")
            seed: Random seed for reproducibility (optional)
            style: Style modifier (optional, e.g., "photorealistic", "anime", "oil painting")
            overlay_text: Text to overlay on image (optional)
        
        Returns:
            Path to generated image file
        """
        # Load model if not loaded
        if not self.model_loaded:
            if not self.load_model():
                raise RuntimeError("Could not load image generation model")
        
        # Parse size
        width, height = self._parse_size(size)
        
        # Enhance prompt with style if provided
        enhanced_prompt = prompt
        if style:
            style_prompts = {
                "photorealistic": "photorealistic, highly detailed, 8k",
                "anime": "anime style, vibrant colors, detailed",
                "oil painting": "oil painting, artistic, detailed brushstrokes",
                "watercolor": "watercolor painting, soft colors, artistic",
                "sketch": "pencil sketch, black and white, detailed",
                "3d render": "3d render, cgi, highly detailed",
            }
            style_text = style_prompts.get(style.lower(), style)
            enhanced_prompt = f"{prompt}, {style_text}"
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        
        print(f"Generating image: {prompt[:50]}...")
        if self.device == "cpu":
            print("Using CPU mode - this may take 1-3 minutes...")
        
        # Generate image
        with torch.no_grad():
            result = self.pipe(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=20 if self.device == "cpu" else 30,
                guidance_scale=7.5,
            )
        
        image = result.images[0]
        
        # Add overlay text if provided
        if overlay_text:
            image = self._add_text_overlay(image, overlay_text)
        
        # Save image
        filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / filename
        image.save(output_path, "PNG")
        
        print(f"✓ Image saved: {output_path}")
        return output_path
    
    def _parse_size(self, size: str) -> tuple:
        """Parse size string to width and height."""
        try:
            parts = size.lower().split('x')
            if len(parts) == 2:
                width = int(parts[0].strip())
                height = int(parts[1].strip())
                # Ensure valid sizes (must be multiple of 8 for Stable Diffusion)
                width = (width // 8) * 8
                height = (height // 8) * 8
                return max(256, min(width, 1024)), max(256, min(height, 1024))
        except:
            pass
        # Default size
        return 512, 512
    
    def _add_text_overlay(self, image: Image.Image, text: str) -> Image.Image:
        """Add text overlay to image."""
        from PIL import ImageDraw, ImageFont
        
        # Create a copy to draw on
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Try to use a nice font, fallback to default
        try:
            font_size = max(20, min(img_with_text.width, img_with_text.height) // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (bottom center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (img_with_text.width - text_width) // 2
        y = img_with_text.height - text_height - 20
        
        # Draw text with outline for visibility
        draw.text((x-1, y-1), text, font=font, fill="black")
        draw.text((x+1, y+1), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")
        
        return img_with_text

