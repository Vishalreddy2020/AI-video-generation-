"""
Image Generation Executor using OpenVINO
Optimized for Intel Arc GPU and CPU
"""
import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import uuid


class ImageGenerateExecutor:
    """
    Image generation executor using OpenVINO for Intel Arc optimization.
    Uses Stable Diffusion 1.5 at 512x512 for reliability.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._detect_device()
        self.pipe = None
        self.model_loaded = False
        
        print(f"Image Generation Executor initialized on device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect the best available device for Intel Arc."""
        # Check for Intel Arc GPU (XPU)
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass
        
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        # Fallback to CPU
        return "cpu"
    
    def _load_model(self):
        """Load the OpenVINO diffusion pipeline."""
        if self.model_loaded:
            return True
        
        try:
            # Try OpenVINO first (best for Intel Arc)
            try:
                from optimum.intel import OVStableDiffusionPipeline
                print("Loading OpenVINO Stable Diffusion model...")
                print("This may take a few minutes on first run (downloading ~2GB model)...")
                
                # Use Stable Diffusion 1.5 at 512x512
                model_id = "runwayml/stable-diffusion-v1-5"
                
                self.pipe = OVStableDiffusionPipeline.from_pretrained(
                    model_id,
                    export=True,  # Export to OpenVINO format if needed
                    device="CPU"  # OpenVINO uses CPU backend (optimized for Intel)
                )
                
                print("✓ OpenVINO model loaded successfully!")
                self.model_loaded = True
                return True
                
            except ImportError:
                print("OpenVINO not available, falling back to standard diffusers...")
                # Fallback to standard diffusers with Intel Arc optimization
                from diffusers import StableDiffusionPipeline
                
                print("Loading Stable Diffusion model (Intel Arc optimized)...")
                print("This may take a few minutes on first run (downloading ~4GB model)...")
                
                model_id = "runwayml/stable-diffusion-v1-5"
                use_fp16 = self.device in ["cuda", "xpu", "mps"]
                
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if use_fp16 else torch.float32,
                    variant="fp16" if use_fp16 else None,
                )
                
                # Move to device
                if self.device == "xpu":
                    try:
                        import intel_extension_for_pytorch as ipex
                        self.pipe = self.pipe.to("xpu")
                        # Optimize for Intel Arc
                        self.pipe = ipex.optimize(self.pipe, dtype=torch.float16)
                    except:
                        self.pipe = self.pipe.to("cpu")
                        self.pipe.enable_model_cpu_offload()
                        self.pipe.enable_attention_slicing()
                elif self.device == "cuda":
                    self.pipe = self.pipe.to("cuda")
                elif self.device == "mps":
                    self.pipe = self.pipe.to("mps")
                else:
                    self.pipe = self.pipe.to("cpu")
                    self.pipe.enable_model_cpu_offload()
                    self.pipe.enable_attention_slicing()
                
                print("✓ Model loaded successfully!")
                self.model_loaded = True
                return True
            
        except ImportError as e:
            print(f"Required libraries not installed: {e}")
            print("Install with: pip install diffusers transformers accelerate")
            print("For OpenVINO: pip install optimum[openvino]")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Path:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            width: Image width (default: 512)
            height: Image height (default: 512)
            num_inference_steps: Number of denoising steps (default: 20)
            guidance_scale: Guidance scale (default: 7.5)
            seed: Random seed for reproducibility (optional)
        
        Returns:
            Path to generated image file
        """
        # Load model if not loaded
        if not self.model_loaded:
            if not self._load_model():
                raise RuntimeError("Could not load image generation model")
        
        # Ensure dimensions are valid (must be multiple of 8 for Stable Diffusion)
        width = (width // 8) * 8
        height = (height // 8) * 8
        width = max(256, min(width, 1024))
        height = max(256, min(height, 1024))
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device if self.device != "xpu" else "cpu")
            generator.manual_seed(seed)
        
        print(f"Generating image: '{prompt[:50]}...'")
        print(f"Size: {width}x{height}, Steps: {num_inference_steps}")
        if self.device == "cpu":
            print("Using CPU mode - this may take 1-3 minutes...")
            num_inference_steps = min(num_inference_steps, 20)  # Fewer steps on CPU
        
        # Generate image
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        image = result.images[0]
        
        # Save image
        filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / filename
        image.save(output_path, "PNG")
        
        print(f"✓ Image saved: {output_path}")
        return output_path

