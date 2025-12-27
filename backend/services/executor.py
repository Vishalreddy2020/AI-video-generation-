"""
Executor Service
Executes plans created by the Planner.
Routes to appropriate model pipelines based on plan.
"""
import json
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image


class Executor:
    """
    Executes plans by routing to appropriate services.
    """
    
    def __init__(self):
        self.image_generator = None
        self.image_editor = None
        self.video_generator = None
        
        # Lazy load services
        self._load_services()
    
    def _load_services(self):
        """Lazy load services to avoid import errors if not installed."""
        # Image generation
        try:
            from services.image_generation import ImageGenerationService
            self.image_generator = ImageGenerationService()
        except ImportError:
            print("ImageGenerationService not available")
        
        # Image editing
        try:
            from services.image_editing import ImageEditingService
            self.image_editor = ImageEditingService()
        except ImportError:
            print("ImageEditingService not available")
        
        # Video generation
        try:
            from services.video_generator import VideoGenerator
            self.video_generator = VideoGenerator()
        except ImportError:
            print("VideoGenerator not available")
    
    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plan and return results.
        
        Args:
            plan: Plan dict from Planner
        
        Returns:
            Dict with:
            - success: bool
            - output_path: Path to generated file
            - output_type: "image" | "video"
            - error: Error message if failed
        """
        action_type = plan.get("action_type")
        parameters = plan.get("parameters", {})
        input_files = plan.get("input_files", {})
        
        try:
            if action_type == "generate_image":
                return await self._execute_generate_image(parameters)
            
            elif action_type == "edit_image":
                input_path = input_files.get("input")
                if not input_path:
                    return {
                        "success": False,
                        "error": "No input image provided for editing"
                    }
                return await self._execute_edit_image(Path(input_path), parameters)
            
            elif action_type == "generate_video":
                input_path = input_files.get("input")
                return await self._execute_generate_video(
                    Path(input_path) if input_path else None,
                    parameters
                )
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image generation."""
        if not self.image_generator:
            return {
                "success": False,
                "error": "Image generation service not available"
            }
        
        try:
            image = self.image_generator.generate_image(
                prompt=params.get("prompt", ""),
                size=params.get("size", "512x512"),
                seed=params.get("seed"),
                style=params.get("style")
            )
            
            # Add overlay text if provided
            if params.get("overlay_text"):
                image = self.image_generator.overlay_text(
                    image=image,
                    text=params["overlay_text"],
                    position=params.get("position", "bottom_center"),
                    font_size=params.get("font_size")
                )
            
            # Save image
            import uuid
            filename = f"generated_{uuid.uuid4().hex[:8]}.png"
            output_path = Path("outputs/images") / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, "PNG")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "output_type": "image"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Image generation failed: {str(e)}"
            }
    
    async def _execute_edit_image(
        self,
        input_path: Path,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute image editing with mask-based inpainting."""
        if not self.image_editor:
            return {
                "success": False,
                "error": "Image editing service not available"
            }
        
        try:
            # Load input image
            image = Image.open(input_path).convert("RGB")
            
            # Generate mask if mask_prompt is provided
            mask = None
            if params.get("mask_prompt"):
                mask = await self._generate_mask(image, params["mask_prompt"])
            
            # Edit image
            edited_image = self.image_editor.edit_image(
                image=image,
                prompt=params.get("prompt", ""),
                mask=mask,
                strength=params.get("strength", 0.75)
            )
            
            # Save edited image
            import uuid
            filename = f"edited_{uuid.uuid4().hex[:8]}.png"
            output_path = Path("outputs/images") / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            edited_image.save(output_path, "PNG")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "output_type": "image"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Image editing failed: {str(e)}"
            }
    
    async def _execute_generate_video(
        self,
        input_path: Optional[Path],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute video generation."""
        if not self.video_generator:
            return {
                "success": False,
                "error": "Video generation service not available"
            }
        
        try:
            output_path = await self.video_generator.generate(
                input_path=input_path,
                text_prompt=params.get("prompt"),
                duration=params.get("duration", 5)
            )
            
            return {
                "success": True,
                "output_path": str(output_path),
                "output_type": "video"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Video generation failed: {str(e)}"
            }
    
    async def _generate_mask(self, image: Image.Image, mask_prompt: str) -> Optional[Image.Image]:
        """
        Generate a mask based on mask_prompt.
        For v1, this is a simple implementation - can be enhanced with SAM or other segmentation models.
        """
        # Simple implementation: create a mask based on keywords
        # In production, you'd use SAM (Segment Anything Model) or similar
        
        # For now, return None to edit entire image
        # TODO: Implement automatic mask generation using SAM or similar
        return None

