"""
Advanced AI Video Generator using real generative models.
Supports: Stable Video Diffusion, AnimateDiff, and API-based solutions.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import subprocess
import requests
import json
import os
from PIL import Image
import tempfile
import base64
import io

class AdvancedAIVideoGenerator:
    """
    High-level AI video generator using actual generative models.
    Can make photos "perform" actions described in prompts.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # API configurations
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.runway_api_key = os.getenv("RUNWAY_API_KEY")
        self.replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
        
        # Try to load local models
        self.local_model = None
        self._init_local_model()
    
    def _init_local_model(self):
        """Initialize local AI model if available."""
        try:
            # Try to import and load Stable Video Diffusion or similar
            # This is a placeholder - actual implementation would load the model
            pass
        except Exception as e:
            print(f"Local model not available: {e}")
    
    async def generate_from_image_and_prompt(
        self,
        image_path: Path,
        text_prompt: str,
        duration: int = 5
    ) -> Path:
        """
        Generate high-quality video from image and prompt using AI.
        Makes the photo actually perform the described actions.
        """
        import asyncio
        
        # Check if local model dependencies are available (quick check)
        try:
            import diffusers
            has_diffusers = True
        except ImportError:
            has_diffusers = False
            print("AI models not installed - using fast fallback generation")
        
        # Skip slow AI model attempts if not installed - go straight to fast fallback
        if not has_diffusers:
            return await self._generate_enhanced_fallback(image_path, text_prompt, duration)
        
        # Try different methods with timeout
        methods = [
            (self._generate_with_local_model, 30),  # 30 second timeout
            (self._generate_with_replicate_api, 10),
            (self._generate_with_stability_api, 10),
        ]
        
        for method, timeout in methods:
            try:
                result = await asyncio.wait_for(
                    method(image_path, text_prompt, duration),
                    timeout=timeout
                )
                if result:
                    return result
            except asyncio.TimeoutError:
                print(f"Method {method.__name__} timed out after {timeout}s, trying next...")
                continue
            except Exception as e:
                print(f"Method {method.__name__} failed: {e}")
                continue
        
        # Fallback to enhanced generation
        print("Using enhanced fallback generation (fast mode)")
        return await self._generate_enhanced_fallback(image_path, text_prompt, duration)
    
    async def _generate_with_replicate_api(
        self,
        image_path: Path,
        text_prompt: str,
        duration: int
    ) -> Optional[Path]:
        """
        Use Replicate API for high-quality video generation.
        Supports models like: stable-video-diffusion, animate-diff, etc.
        """
        if not self.replicate_api_key:
            return None
        
        try:
            import replicate
            
            # Encode image to base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Use Stable Video Diffusion model
            output = replicate.run(
                "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b81724346",
                input={
                    "image": f"data:image/png;base64,{image_data}",
                    "motion_bucket_id": 127,
                    "cond_aug": 0.02,
                }
            )
            
            # Download and save video
            if output:
                video_url = output[0] if isinstance(output, list) else output
                return await self._download_video(video_url, duration)
        except ImportError:
            print("Replicate not installed. Install with: pip install replicate")
        except Exception as e:
            print(f"Replicate API error: {e}")
        
        return None
    
    async def _generate_with_stability_api(
        self,
        image_path: Path,
        text_prompt: str,
        duration: int
    ) -> Optional[Path]:
        """
        Use Stability AI API for video generation.
        """
        if not self.stability_api_key:
            return None
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Stability AI video generation endpoint
            url = "https://api.stability.ai/v2alpha/generation/image-to-video"
            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "image": image_data,
                "prompt": text_prompt,
                "seed": 0,
                "cfg_scale": 1.8,
                "motion_bucket_id": 127
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'id' in result:
                    # Poll for completion
                    generation_id = result['id']
                    video_url = await self._poll_stability_generation(generation_id)
                    if video_url:
                        return await self._download_video(video_url, duration)
        except Exception as e:
            print(f"Stability API error: {e}")
        
        return None
    
    async def _poll_stability_generation(self, generation_id: str, max_attempts: 30) -> Optional[str]:
        """Poll Stability AI for generation completion."""
        url = f"https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}"
        headers = {"Authorization": f"Bearer {self.stability_api_key}"}
        
        import asyncio
        for _ in range(max_attempts):
            await asyncio.sleep(2)
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                if result.get('finish_reason') == 'SUCCESS':
                    return result.get('video')
            elif response.status_code == 202:
                continue  # Still processing
            else:
                break
        
        return None
    
    async def _generate_with_local_model(
        self,
        image_path: Path,
        text_prompt: str,
        duration: int
    ) -> Optional[Path]:
        """
        Use local AI model (Stable Video Diffusion, AnimateDiff, etc.)
        Requires model files and GPU.
        """
        try:
            from .local_ai_model import LocalAIVideoModel
            
            # Quick check - if model loading takes too long, skip
            model = LocalAIVideoModel()
            
            # Try to load model with quick timeout check
            if not model.model_loaded:
                print("Attempting to load local AI model...")
                # Don't wait long for model loading
                loaded = model.load_model("svd")
                if not loaded:
                    print("Local model not available, skipping")
                    return None
            
            # Calculate frames based on duration (8 fps for SVD)
            fps = 8
            num_frames = duration * fps
            
            # Generate video
            output_path = await model.generate_video(
                image_path=image_path,
                text_prompt=text_prompt,
                num_frames=num_frames,
                num_inference_steps=25,
            )
            
            return output_path
            
        except ImportError:
            # Silently skip if dependencies not installed
            return None
        except Exception as e:
            print(f"Local model generation failed: {e}")
            return None
    
    async def _generate_enhanced_fallback(
        self,
        image_path: Path,
        text_prompt: str,
        duration: int
    ) -> Path:
        """
        Enhanced fallback that creates more realistic animations
        when AI models aren't available.
        """
        from .ai_video_generator import AIVideoGenerator
        basic_generator = AIVideoGenerator()
        return await basic_generator.generate_from_image_and_prompt(
            image_path, text_prompt, duration
        )
    
    async def _download_video(self, video_url: str, duration: int) -> Path:
        """Download video from URL and save locally."""
        response = requests.get(video_url, stream=True)
        output_path = self.output_dir / f"ai_generated_{duration}s.mp4"
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Trim to desired duration if needed
        final_path = self.output_dir / f"final_ai_{duration}s.mp4"
        self._trim_video(output_path, final_path, duration)
        
        return final_path
    
    def _trim_video(self, input_path: Path, output_path: Path, duration: int):
        """Trim video to specified duration."""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-t', str(duration),
                '-c', 'copy',
                '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except:
            import shutil
            shutil.copy(input_path, output_path)

