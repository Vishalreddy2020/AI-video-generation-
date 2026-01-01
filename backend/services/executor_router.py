"""
Executor Router
Maps plan operations to executors and chains them together.
This is what makes the app feel "smart" - it can execute complex multi-step plans.
"""
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import io

from schemas.plan import Plan, Operation
from services.executors.image_generate import ImageGenerateExecutor
from services.executors.image_edit_inpaint import ImageEditInpaintExecutor
from services.image_generation import ImageGenerationService

logger = logging.getLogger(__name__)


class ExecutorRouter:
    """
    Routes plan operations to appropriate executors and chains them together.
    """
    
    def __init__(self):
        self.image_generate_executor = ImageGenerateExecutor()
        self.image_edit_inpaint_executor = ImageEditInpaintExecutor()
        self.image_generation_service = ImageGenerationService()
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def execute_plan(
        self,
        plan: Plan,
        input_image: Optional[Image.Image] = None,
        input_mask: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan end-to-end, chaining operations together.
        
        Args:
            plan: Plan object with ops to execute
            input_image: Optional initial image (from uploaded file)
            input_mask: Optional initial mask (from uploaded file)
        
        Returns:
            Dict with:
            - success: bool
            - output_path: Path to final asset
            - output_type: "image" | "video"
            - job_id: Unique job identifier
            - metadata: Execution metadata (seed, ops_ran, etc.)
            - error: Error message if failed
        """
        job_id = str(uuid.uuid4())
        ops_ran = []
        current_image = input_image
        used_seed = plan.params.get("seed")
        
        try:
            # Validate plan
            self._validate_plan(plan)
            
            # Execute each operation in sequence
            for i, op in enumerate(plan.ops):
                logger.info(f"Executing operation {i+1}/{len(plan.ops)}: {op.op_type}")
                
                # Execute operation
                result = self._execute_operation(
                    op=op,
                    current_image=current_image,
                    current_mask=input_mask if i == 0 else None,  # Only use input mask for first op
                    plan_params=plan.params,
                    job_id=job_id
                )
                
                # Update current image for next operation
                if result.get("image"):
                    current_image = result["image"]
                
                # Track operation
                ops_ran.append({
                    "op_type": op.op_type,
                    "target": op.target,
                    "success": result.get("success", True)
                })
            
            # If no operations but we have an input image, return it
            if not plan.ops and current_image:
                output_path = self._save_image(current_image, job_id)
                return {
                    "success": True,
                    "output_path": output_path,
                    "output_type": "image",
                    "job_id": job_id,
                    "metadata": {
                        "ops_ran": ops_ran,
                        "seed": used_seed,
                        "task_type": plan.task_type
                    }
                }
            
            # If no operations and no input, generate default
            if not plan.ops and not current_image:
                if plan.task_type == "image_generate":
                    # Generate image from prompt
                    prompt = plan.params.get("prompt", "a beautiful image")
                    current_image = self._generate_image(prompt, plan.params)
                    output_path = self._save_image(current_image, job_id)
                    return {
                        "success": True,
                        "output_path": output_path,
                        "output_type": "image",
                        "job_id": job_id,
                        "metadata": {
                            "ops_ran": [{"op_type": "generate", "success": True}],
                            "seed": used_seed,
                            "task_type": plan.task_type
                        }
                    }
            
            # Save final image
            if current_image:
                output_path = self._save_image(current_image, job_id)
                return {
                    "success": True,
                    "output_path": output_path,
                    "output_type": "image",
                    "job_id": job_id,
                    "metadata": {
                        "ops_ran": ops_ran,
                        "seed": used_seed,
                        "task_type": plan.task_type,
                        "num_ops": len(ops_ran)
                    }
                }
            else:
                raise RuntimeError("No output generated from plan")
        
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id,
                "metadata": {
                    "ops_ran": ops_ran,
                    "task_type": plan.task_type
                }
            }
    
    def _validate_plan(self, plan: Plan):
        """Validate plan structure."""
        if not plan.task_type:
            raise ValueError("Plan missing task_type")
        
        # Validate operations
        for op in plan.ops:
            if not op.op_type:
                raise ValueError(f"Operation missing op_type: {op}")
            
            valid_op_types = [
                "inpaint", "outpaint", "overlay_text", "upscale",
                "generate", "edit", "filter", "transform"
            ]
            if op.op_type not in valid_op_types:
                raise ValueError(f"Invalid op_type: {op.op_type}. Must be one of {valid_op_types}")
    
    def _execute_operation(
        self,
        op: Operation,
        current_image: Optional[Image.Image],
        current_mask: Optional[Image.Image],
        plan_params: Dict[str, Any],
        job_id: str
    ) -> Dict[str, Any]:
        """Execute a single operation."""
        
        if op.op_type == "generate":
            return self._execute_generate(op, plan_params)
        
        elif op.op_type == "inpaint":
            if not current_image:
                raise ValueError("inpaint operation requires an input image")
            return self._execute_inpaint(op, current_image, current_mask, plan_params)
        
        elif op.op_type == "overlay_text":
            if not current_image:
                raise ValueError("overlay_text operation requires an input image")
            return self._execute_overlay_text(op, current_image, plan_params)
        
        elif op.op_type == "upscale":
            if not current_image:
                raise ValueError("upscale operation requires an input image")
            return self._execute_upscale(op, current_image, plan_params)
        
        elif op.op_type == "edit":
            if not current_image:
                raise ValueError("edit operation requires an input image")
            # Generic edit - use inpaint with auto mask
            return self._execute_edit(op, current_image, plan_params)
        
        else:
            raise ValueError(f"Unsupported operation type: {op.op_type}")
    
    def _execute_generate(self, op: Operation, plan_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generate operation."""
        prompt = op.prompt or plan_params.get("prompt", "a beautiful image")
        size = plan_params.get("size", "512x512")
        width, height = map(int, size.split("x"))
        
        # Generate image
        output_path = self.image_generate_executor.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=plan_params.get("steps", 20),
            guidance_scale=plan_params.get("guidance_scale", 7.5),
            seed=plan_params.get("seed")
        )
        
        # Load generated image
        image = Image.open(output_path).convert("RGB")
        
        return {
            "success": True,
            "image": image,
            "output_path": output_path
        }
    
    def _execute_inpaint(
        self,
        op: Operation,
        image: Image.Image,
        mask: Optional[Image.Image],
        plan_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute inpaint operation."""
        prompt = op.prompt or plan_params.get("prompt", "edit the image")
        strength = plan_params.get("strength", 0.75)
        
        # If no mask provided and target specified, we'd need auto mask generation
        # For now, use None mask (edits entire image if no mask)
        # TODO: Integrate with image_edit_auto for automatic mask generation
        
        output_path = self.image_edit_inpaint_executor.edit(
            image=image,
            mask=mask,
            prompt=prompt,
            strength=strength
        )
        
        edited_image = Image.open(output_path).convert("RGB")
        
        return {
            "success": True,
            "image": edited_image,
            "output_path": output_path
        }
    
    def _execute_overlay_text(
        self,
        op: Operation,
        image: Image.Image,
        plan_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute overlay_text operation."""
        text = op.params.get("text") or op.prompt
        if not text:
            raise ValueError("overlay_text operation requires text parameter")
        
        position = op.params.get("position", "bottom_center")
        font_size = op.params.get("font_size")
        
        # Use ImageGenerationService overlay_text method
        image_with_text = self.image_generation_service.overlay_text(
            image=image,
            text=text,
            position=position,
            font_size=font_size
        )
        
        return {
            "success": True,
            "image": image_with_text
        }
    
    def _execute_upscale(
        self,
        op: Operation,
        image: Image.Image,
        plan_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute upscale operation."""
        scale = op.params.get("scale", 2)
        
        # Simple upscale using PIL (bicubic resampling)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return {
            "success": True,
            "image": upscaled_image
        }
    
    def _execute_edit(
        self,
        op: Operation,
        image: Image.Image,
        plan_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute generic edit operation (uses inpaint with no mask)."""
        prompt = op.prompt or plan_params.get("prompt", "edit the image")
        strength = plan_params.get("strength", 0.75)
        
        # Use inpaint without mask (edits entire image)
        output_path = self.image_edit_inpaint_executor.edit(
            image=image,
            mask=None,
            prompt=prompt,
            strength=strength
        )
        
        edited_image = Image.open(output_path).convert("RGB")
        
        return {
            "success": True,
            "image": edited_image,
            "output_path": output_path
        }
    
    def _generate_image(self, prompt: str, params: Dict[str, Any]) -> Image.Image:
        """Generate image from prompt (helper method)."""
        size = params.get("size", "512x512")
        width, height = map(int, size.split("x"))
        
        output_path = self.image_generate_executor.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=params.get("steps", 20),
            guidance_scale=params.get("guidance_scale", 7.5),
            seed=params.get("seed")
        )
        
        return Image.open(output_path).convert("RGB")
    
    def _save_image(self, image: Image.Image, job_id: str) -> Path:
        """Save image to disk."""
        filename = f"final_{job_id[:8]}.png"
        output_path = self.output_dir / filename
        image.save(output_path, "PNG")
        return output_path

