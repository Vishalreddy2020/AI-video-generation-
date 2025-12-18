"""
Local AI Video Generation Model
Uses open-source models like Stable Video Diffusion or AnimateDiff
No API keys required - runs entirely locally
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import subprocess
import os

class LocalAIVideoModel:
    """
    Local AI model for video generation.
    Can use Stable Video Diffusion, AnimateDiff, or similar open-source models.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Detect Intel Arc GPU and NPU
        self.device = self._detect_device()
        self.model = None
        self.pipe = None
        self.model_loaded = False
        
        print(f"Using device: {self.device}")
        if "xpu" in self.device:
            print("✓ Intel Arc GPU detected! Using GPU acceleration.")
        elif "npu" in self.device:
            print("✓ Intel NPU detected! Using NPU acceleration.")
        elif self.device == "cpu":
            print("Using CPU mode (Intel Arc GPU not detected or not configured)")
    
    def _detect_device(self) -> str:
        """Detect the best available device - works on any laptop."""
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            print("✓ NVIDIA GPU detected!")
            return "cuda"
        
        # Check for Intel Extension for PyTorch (Intel Arc GPU support)
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                print("✓ Intel Arc GPU (XPU) detected!")
                return "xpu"
        except ImportError:
            pass
        except Exception:
            pass
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ Apple Silicon (MPS) detected!")
            return "mps"
        
        # Default to CPU - works on every laptop
        print("Using CPU mode (works on all laptops)")
        return "cpu"
    
    def load_model(self, model_type: str = "svd"):
        """
        Load the AI model.
        
        Args:
            model_type: "svd" for Stable Video Diffusion, "animatediff" for AnimateDiff
        """
        try:
            if model_type == "svd":
                return self._load_stable_video_diffusion()
            elif model_type == "animatediff":
                return self._load_animate_diff()
            else:
                print(f"Unknown model type: {model_type}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simpler generation method")
            return False
    
    def _load_stable_video_diffusion(self) -> bool:
        """Load Stable Video Diffusion model."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video
            import torch
            
            print("Loading Stable Video Diffusion model...")
            print("This may take a few minutes on first run (downloading ~5GB model)...")
            
            # Load the model - optimized for any device
            use_fp16 = self.device in ["cuda", "xpu", "mps"]  # Use FP16 for GPU
            
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                variant="fp16" if use_fp16 else None,
            )
            
            # Move to appropriate device - universal support
            if self.device == "cuda":
                pipe = pipe.to("cuda")
            elif self.device == "xpu":
                try:
                    import intel_extension_for_pytorch as ipex
                    pipe = pipe.to("xpu")
                    print("Model loaded on Intel Arc GPU")
                except:
                    pipe = pipe.to("cpu")
                    pipe.enable_model_cpu_offload()
            elif self.device == "mps":
                pipe = pipe.to("mps")
            else:
                # CPU mode - optimized for laptops
                pipe = pipe.to("cpu")
                # Use CPU offloading to save memory on laptops
                pipe.enable_model_cpu_offload()
                # Enable attention slicing for better CPU performance
                pipe.enable_attention_slicing()
            
            self.pipe = pipe
            self.model_loaded = True
            print("✓ Model loaded successfully!")
            return True
            
        except ImportError:
            print("diffusers library not installed.")
            print("Install with: pip install diffusers transformers accelerate")
            return False
        except Exception as e:
            print(f"Error loading Stable Video Diffusion: {e}")
            return False
    
    def _load_animate_diff(self) -> bool:
        """Load AnimateDiff model."""
        try:
            from diffusers import AnimateDiffPipeline, DDIMScheduler
            from diffusers.utils import export_to_video
            import torch
            
            print("Loading AnimateDiff model...")
            
            pipe = AnimateDiffPipeline.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            if self.device == "cuda":
                pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cpu")
                pipe.enable_model_cpu_offload()
            
            self.pipe = pipe
            self.model_loaded = True
            print("✓ AnimateDiff model loaded successfully!")
            return True
            
        except ImportError:
            print("diffusers library not installed.")
            return False
        except Exception as e:
            print(f"Error loading AnimateDiff: {e}")
            return False
    
    async def generate_video(
        self,
        image_path: Path,
        text_prompt: Optional[str] = None,
        num_frames: int = 25,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5
    ) -> Path:
        """
        Generate video from image using the loaded AI model.
        
        Args:
            image_path: Path to input image
            text_prompt: Optional text prompt (for some models)
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
        """
        if not self.model_loaded:
            # Try to load model automatically
            if not self.load_model("svd"):
                raise RuntimeError("Could not load AI model. Please install dependencies.")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Resize image to model's expected size
            image = image.resize((1024, 576))  # SVD standard size
            
            print(f"Generating {num_frames} frames...")
            print("This may take 1-5 minutes depending on your hardware...")
            
            # Generate video frames
            if hasattr(self.pipe, 'generate'):
                # Stable Video Diffusion
                frames = self.pipe(
                    image,
                    decode_chunk_size=8,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    motion_bucket_id=127,
                    noise_aug_strength=0.02,
                ).frames[0]
            else:
                # AnimateDiff or other models
                frames = self.pipe(
                    prompt=text_prompt or "smooth motion, high quality",
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).frames[0]
            
            # Export to video
            output_path = self.output_dir / "ai_generated_video.mp4"
            self._frames_to_video(frames, output_path, fps=8)
            
            print(f"✓ Video generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating video: {e}")
            raise
    
    def _frames_to_video(self, frames, output_path: Path, fps: int = 8):
        """Convert frames to video file."""
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, str(output_path), fps=fps)
        except:
            # Fallback: use OpenCV
            if len(frames) == 0:
                raise ValueError("No frames generated")
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()

