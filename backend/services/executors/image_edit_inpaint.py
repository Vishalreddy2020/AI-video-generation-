"""
Image Edit Inpaint Executor
High-level local image editing using Stable Diffusion Inpainting.

Takes: image + mask + prompt → edited image (mask-based control).
"""
from pathlib import Path
from typing import Optional
import uuid

from PIL import Image

from services.image_editing import ImageEditingService


class ImageEditInpaintExecutor:
    """
    Wrapper around ImageEditingService for mask-based inpainting.
    """

    def __init__(self) -> None:
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.service = ImageEditingService()

    def edit(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
        prompt: str,
        strength: float,
    ) -> Path:
        """
        Run mask-based inpainting and save result to disk.

        Args:
            image: Source PIL image (RGB).
            mask: Optional mask image (L mode). White = edit, black = keep.
            prompt: Edit description.
            strength: 0.0–1.0, recommended 0.6–0.9 for stronger edits.

        Returns:
            Path to edited PNG image.
        """
        edited = self.service.edit_image(
            image=image,
            prompt=prompt,
            mask=mask,
            strength=strength,
        )

        filename = f"edited_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / filename
        edited.save(output_path, "PNG")
        return output_path


