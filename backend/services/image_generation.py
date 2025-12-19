"""
Image Generation Service
Generates images from text prompts using Stable Diffusion.
"""
import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import uuid

class ImageGenerationService:
    """
    Service for generating images from text prompts.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._detect_device()
        self.pipe = None
        self.model_loaded = False
        
        print(f"Image Generation Service initialized on device: {self.device}")
    
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
    
    def generate_image(
        self,
        prompt: str,
        size: str = "512x512",
        seed: Optional[int] = None,
        style: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size (e.g., "512x512", "768x768", "1024x1024")
            seed: Random seed for reproducibility (optional)
            style: Style modifier (optional: "photorealistic", "anime", "oil painting", etc.)
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale (default: 7.5)
        
        Returns:
            PIL.Image: Generated image
        """
        # Load model if not loaded
        if not self.model_loaded:
            if not self._load_model():
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
                "cartoon": "cartoon style, vibrant, animated",
                "cyberpunk": "cyberpunk style, neon lights, futuristic",
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
            num_inference_steps = min(num_inference_steps, 20)  # Fewer steps on CPU
        
        # Generate image
        with torch.no_grad():
            result = self.pipe(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        
        image = result.images[0]
        return image
    
    def overlay_text(
        self,
        image: Image.Image,
        text: str,
        position: str = "bottom_center",
        font_size: Optional[int] = None,
        font_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 2
    ) -> Image.Image:
        """
        Overlay text on an image.
        
        Args:
            image: PIL Image to overlay text on
            text: Text to overlay
            position: Position of text ("top_left", "top_center", "top_right",
                      "center_left", "center", "center_right",
                      "bottom_left", "bottom_center", "bottom_right")
            font_size: Font size (auto-calculated if None)
            font_color: Text color (default: "white")
            outline_color: Outline color for text (default: "black")
            outline_width: Outline width in pixels (default: 2)
        
        Returns:
            PIL.Image: Image with text overlay
        """
        # Create a copy to draw on
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Calculate font size if not provided
        if font_size is None:
            font_size = max(20, min(img_with_text.width, img_with_text.height) // 20)
        
        # Try to load a nice font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Calculate text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        x, y = self._calculate_text_position(
            position, img_with_text.width, img_with_text.height,
            text_width, text_height
        )
        
        # Draw text outline (for visibility)
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=font_color)
        
        return img_with_text
    
    def _parse_size(self, size: str) -> Tuple[int, int]:
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
    
    def _calculate_text_position(
        self,
        position: str,
        img_width: int,
        img_height: int,
        text_width: int,
        text_height: int,
        margin: int = 20
    ) -> Tuple[int, int]:
        """Calculate text position based on position string."""
        positions = {
            "top_left": (margin, margin),
            "top_center": ((img_width - text_width) // 2, margin),
            "top_right": (img_width - text_width - margin, margin),
            "center_left": (margin, (img_height - text_height) // 2),
            "center": ((img_width - text_width) // 2, (img_height - text_height) // 2),
            "center_right": (img_width - text_width - margin, (img_height - text_height) // 2),
            "bottom_left": (margin, img_height - text_height - margin),
            "bottom_center": ((img_width - text_width) // 2, img_height - text_height - margin),
            "bottom_right": (img_width - text_width - margin, img_height - text_height - margin),
        }
        return positions.get(position.lower(), positions["bottom_center"])

