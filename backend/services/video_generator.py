import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
from PIL import Image

# Try to import torch (optional, for future AI model support)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import AI video generators
try:
    from .ai_video_generator import AIVideoGenerator
    AI_GENERATOR_AVAILABLE = True
except ImportError:
    AI_GENERATOR_AVAILABLE = False

try:
    from .advanced_ai_generator import AdvancedAIVideoGenerator
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    ADVANCED_AI_AVAILABLE = False

class VideoGenerator:
    """
    Generates video from photo, text, or video input.
    Uses AI models for video generation.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        # Check for CUDA if torch is available, otherwise default to CPU
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        # Initialize AI video generators
        if ADVANCED_AI_AVAILABLE:
            self.ai_generator = AdvancedAIVideoGenerator()
            self.use_advanced = True
        elif AI_GENERATOR_AVAILABLE:
            self.ai_generator = AIVideoGenerator()
            self.use_advanced = False
        else:
            self.ai_generator = None
            self.use_advanced = False
    
    async def generate(
        self,
        input_path: Optional[Path] = None,
        text_prompt: Optional[str] = None,
        duration: int = 5
    ) -> Path:
        """
        Generate video from input.
        
        Args:
            input_path: Path to input image/video
            text_prompt: Text description for video generation
            duration: Video duration in seconds (5 or 10)
        
        Returns:
            Path to generated video file
        """
        # Determine number of frames (assuming 24 fps)
        fps = 24
        total_frames = duration * fps
        
        if input_path and input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # If we have both image and text prompt, try AI generator with timeout
            if text_prompt and self.ai_generator:
                import asyncio
                try:
                    # Try AI generation with 5 second timeout to avoid hanging
                    result = await asyncio.wait_for(
                        self.ai_generator.generate_from_image_and_prompt(
                            input_path, text_prompt, duration
                        ),
                        timeout=5.0  # Quick timeout - if AI models not ready, skip fast
                    )
                    return result
                except asyncio.TimeoutError:
                    print("AI generation timeout - using fast standard generation")
                except Exception as e:
                    print(f"AI generation failed: {e}, using standard generation")
            # Use standard image animation (fast and reliable)
            return await self._generate_from_image(input_path, duration, fps, total_frames, text_prompt)
        elif input_path and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Process existing video (with optional text instructions)
            return await self._process_video(input_path, duration, fps, text_prompt)
        elif text_prompt:
            # Generate video from text
            return await self._generate_from_text(text_prompt, duration, fps, total_frames)
        else:
            raise ValueError("No valid input provided")
    
    async def _generate_from_image(
        self,
        image_path: Path,
        duration: int,
        fps: int,
        total_frames: int,
        text_prompt: Optional[str] = None
    ) -> Path:
        """Generate video from a single image with subtle animation.
        
        Args:
            text_prompt: Optional text instructions for video generation
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize to 720p (1280x720)
        height, width = img.shape[:2]
        target_width = 1280
        target_height = 720
        
        # Maintain aspect ratio
        aspect = width / height
        if aspect > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            new_height = target_height
            new_width = int(target_height * aspect)
        
        img = cv2.resize(img, (new_width, new_height))
        
        # Create canvas for 720p
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img
        
        # Adjust animation based on text prompt (if provided)
        animation_intensity = 0.1
        if text_prompt:
            prompt_lower = text_prompt.lower()
            if 'zoom' in prompt_lower or 'close' in prompt_lower:
                animation_intensity = 0.2
            elif 'subtle' in prompt_lower or 'gentle' in prompt_lower:
                animation_intensity = 0.05
            elif 'dramatic' in prompt_lower or 'intense' in prompt_lower:
                animation_intensity = 0.3
        
        # Generate video with subtle zoom/pan animation
        output_path = self.output_dir / f"generated_{duration}s.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
        
        for frame_num in range(total_frames):
            # Create subtle animation (zoom and pan)
            progress = frame_num / total_frames
            zoom = 1.0 + animation_intensity * np.sin(progress * 2 * np.pi)
            pan_x = int(20 * animation_intensity * 10 * np.sin(progress * np.pi))
            pan_y = int(10 * animation_intensity * 10 * np.cos(progress * np.pi))
            
            # Apply transformation
            center_x, center_y = target_width // 2, target_height // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom)
            M[0, 2] += pan_x
            M[1, 2] += pan_y
            
            frame = cv2.warpAffine(canvas, M, (target_width, target_height), 
                                 borderMode=cv2.BORDER_REPLICATE)
            
            out.write(frame)
        
        out.release()
        
        # Re-encode to ensure compatibility
        final_path = self.output_dir / f"final_{duration}s.mp4"
        self._reencode_video(output_path, final_path, fps)
        
        return final_path
    
    async def _process_video(
        self,
        video_path: Path,
        duration: int,
        fps: int,
        text_prompt: Optional[str] = None
    ) -> Path:
        """Process existing video to match duration and resolution."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Resize to 720p
        target_width = 1280
        target_height = 720
        
        output_path = self.output_dir / f"processed_{duration}s.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
        
        total_frames = duration * fps
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (target_width, target_height))
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Re-encode
        final_path = self.output_dir / f"final_{duration}s.mp4"
        self._reencode_video(output_path, final_path, fps)
        
        return final_path
    
    async def _generate_from_text(
        self,
        text_prompt: str,
        duration: int,
        fps: int,
        total_frames: int
    ) -> Path:
        """Generate video from text prompt."""
        # In production, this would use a text-to-video model like:
        # - Stable Video Diffusion
        # - AnimateDiff
        # - ModelScope
        # For now, create a placeholder video
        
        target_width = 1280
        target_height = 720
        
        output_path = self.output_dir / f"text_generated_{duration}s.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
        
        # Create animated gradient based on text
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Create animated gradient
            for y in range(target_height):
                for x in range(target_width):
                    r = int(255 * (x / target_width + progress) % 1)
                    g = int(255 * (y / target_height + progress) % 1)
                    b = int(255 * progress)
                    frame[y, x] = [b, g, r]
            
            out.write(frame)
        
        out.release()
        
        # Re-encode
        final_path = self.output_dir / f"final_{duration}s.mp4"
        self._reencode_video(output_path, final_path, fps)
        
        return final_path
    
    def _reencode_video(self, input_path: Path, output_path: Path, fps: int):
        """Re-encode video using ffmpeg for better compatibility."""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-r', str(fps),
                '-pix_fmt', 'yuv420p',
                '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg is not available, just copy the file
            import shutil
            shutil.copy(input_path, output_path)

