"""
Image Edit Auto Executor
Automatic image editing where the AI determines what to edit based on the prompt.
Uses auto mask generation + inpainting.

Pipeline:
1. Extract target object from edit_prompt
2. Generate mask using segmentation
3. Run inpainting with generated mask
"""
import re
import io
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from services.image_editing import ImageEditingService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEditAutoExecutor:
    """
    Executor for automatic image editing with AI-determined mask generation.
    """

    def __init__(self):
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.editing_service = ImageEditingService()
        self.segmentation_model = None
        self.segmentation_loaded = False

    def _extract_target_object(self, edit_prompt: str) -> Optional[str]:
        """
        Extract the target object/region from edit prompt.
        
        Examples:
        - "change shirt to black" -> "shirt"
        - "make the sky sunset orange" -> "sky"
        - "remove the logo" -> "logo"
        - "add sunglasses to the person" -> "person"
        
        Returns:
            Extracted target phrase, or None if extraction fails
        """
        prompt_lower = edit_prompt.lower().strip()
        
        # Common patterns for target extraction
        patterns = [
            r"(?:change|edit|modify|make|transform|replace|update)\s+(?:the\s+)?([a-z\s]+?)(?:\s+to|\s+with|\s+into|\s+as|$)",
            r"(?:remove|delete|erase)\s+(?:the\s+)?([a-z\s]+?)(?:\s+from|\s+in|$)",
            r"(?:add|put|place)\s+([a-z\s]+?)(?:\s+to|\s+on|\s+in|\s+over)",
            r"the\s+([a-z\s]+?)(?:\s+should|\s+needs|\s+will|\s+to)",
            r"([a-z\s]+?)(?:\s+should\s+be|\s+needs\s+to|\s+will\s+be)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                target = match.group(1).strip()
                # Filter out common stop words and keep meaningful phrases
                stop_words = {'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'for', 'with'}
                words = [w for w in target.split() if w not in stop_words]
                if words:
                    return ' '.join(words[:3])  # Limit to 3 words max
        
        # Fallback: try to find common objects/regions
        common_targets = [
            'shirt', 'pants', 'dress', 'jacket', 'clothing', 'clothes',
            'sky', 'background', 'foreground', 'ground',
            'face', 'hair', 'eyes', 'person', 'people',
            'logo', 'text', 'sign', 'label',
            'car', 'vehicle', 'building', 'house',
            'tree', 'grass', 'water', 'ocean', 'beach'
        ]
        
        for target in common_targets:
            if target in prompt_lower:
                return target
        
        return None

    def _generate_mask_segmentation(
        self, 
        image: Image.Image, 
        target_phrase: Optional[str]
    ) -> Image.Image:
        """
        Generate mask using segmentation model.
        
        Args:
            image: PIL Image (RGB)
            target_phrase: Optional target phrase (e.g., "shirt", "sky")
        
        Returns:
            PIL Image mask (L mode, white=edit, black=keep)
        """
        width, height = image.size
        
        # Primary: Use OpenCV-based salient object detection (works everywhere)
        try:
            mask = self._generate_mask_salient(image)
            # If we got a reasonable mask (not all black/white), use it
            mask_array = np.array(mask)
            if mask_array.sum() > 0 and mask_array.sum() < (width * height * 255 * 0.95):
                logger.info("Using salient object detection mask")
                return mask
        except Exception as e:
            logger.warning(f"Salient object detection failed: {e}")
        
        # Optional: Try transformers segmentation (if available)
        try:
            mask = self._generate_mask_with_transformers(image, target_phrase)
            mask_array = np.array(mask)
            if mask_array.sum() > 0 and mask_array.sum() < (width * height * 255 * 0.95):
                logger.info("Using transformers segmentation mask")
                return mask
        except Exception as e:
            logger.warning(f"Transformers segmentation failed: {e}")
        
        # Final fallback: center region mask
        logger.info("Using center region fallback mask")
        return self._generate_mask_center(image)

    def _generate_mask_with_transformers(
        self, 
        image: Image.Image, 
        target_phrase: Optional[str]
    ) -> Image.Image:
        """
        Generate mask using transformers segmentation model.
        Uses a lightweight approach that works on CPU.
        """
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError("transformers not available for segmentation")
        
        # Load model on first use
        if not self.segmentation_loaded:
            logger.info("Loading segmentation model...")
            try:
                # Use image-segmentation pipeline with a lightweight model
                # This will download on first use (~100-200MB)
                device = 0 if torch.cuda.is_available() else -1  # -1 = CPU
                self.segmentation_pipe = pipeline(
                    "image-segmentation",
                    model="mattmdjaga/segformer_b2_clothes",  # Lightweight clothes segmentation
                    device=device
                )
                self.segmentation_loaded = True
                logger.info("Segmentation model loaded")
            except Exception as e:
                logger.warning(f"Failed to load clothes segmentation, trying fallback: {e}")
                try:
                    # Fallback to even simpler model
                    device = 0 if torch.cuda.is_available() else -1
                    self.segmentation_pipe = pipeline(
                        "image-segmentation",
                        model="nvidia/segformer-b0-finetuned-ade-640-640",  # General segmentation
                        device=device
                    )
                    self.segmentation_loaded = True
                    logger.info("Fallback segmentation model loaded")
                except Exception as e2:
                    logger.error(f"Failed to load any segmentation model: {e2}")
                    raise
        
        # Process image
        try:
            results = self.segmentation_pipe(image)
            
            # Combine all segments into a single mask
            # For clothes model, we want to get relevant segments
            mask_array = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            
            # If we have target phrase, try to match segments
            if target_phrase:
                target_lower = target_phrase.lower()
                for result in results:
                    label_lower = result['label'].lower()
                    # Check if label matches target
                    if any(word in label_lower for word in target_lower.split()):
                        mask = np.array(result['mask'])
                        mask_array = np.maximum(mask_array, mask)
            
            # If no match or no target, use largest segment
            if mask_array.sum() == 0 and results:
                # Get the largest segment
                largest = max(results, key=lambda x: np.array(x['mask']).sum())
                mask_array = np.array(largest['mask'])
            
            # Convert to binary (white=edit, black=keep)
            mask_array = (mask_array > 127).astype(np.uint8) * 255
            
        except Exception as e:
            logger.warning(f"Segmentation pipeline failed: {e}, using fallback")
            raise
        
        # Convert to PIL
        mask_pil = Image.fromarray(mask_array, mode="L")
        return mask_pil

    def _generate_mask_salient(self, image: Image.Image) -> Image.Image:
        """
        Generate mask using salient object detection (OpenCV-based).
        This is the primary method as it works reliably on any system.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python not available")
        
        # Convert PIL to numpy
        img_array = np.array(image.convert("RGB"))
        height, width = img_array.shape[:2]
        
        # Method 1: Try OpenCV's saliency detection
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(img_array)
            
            if success:
                # Convert saliency map to binary mask
                mask_array = (saliency_map * 255).astype(np.uint8)
                # Use adaptive threshold to get better binary mask
                _, mask_array = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Clean up mask: remove small noise
                kernel = np.ones((3, 3), np.uint8)
                mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
                mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
                
                # If mask is reasonable (not too small/large), use it
                mask_ratio = mask_array.sum() / (width * height * 255)
                if 0.05 < mask_ratio < 0.95:
                    mask_pil = Image.fromarray(mask_array, mode="L")
                    return mask_pil
        except Exception as e:
            logger.warning(f"Saliency detection failed: {e}")
        
        # Method 2: Use GrabCut-like approach (foreground extraction)
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Create a simple foreground mask using thresholding
            # Use adaptive threshold to handle varying lighting
            mask_array = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours and get the largest one (likely the main object)
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Create mask from largest contour
                mask_array = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask_array, [largest_contour], 255)
                
                # Dilate slightly to include edges
                kernel = np.ones((5, 5), np.uint8)
                mask_array = cv2.dilate(mask_array, kernel, iterations=2)
                
                mask_ratio = mask_array.sum() / (width * height * 255)
                if 0.05 < mask_ratio < 0.95:
                    mask_pil = Image.fromarray(mask_array, mode="L")
                    return mask_pil
        except Exception as e:
            logger.warning(f"GrabCut-like detection failed: {e}")
        
        # Method 3: Simple edge-based region detection
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges to create regions
            kernel = np.ones((7, 7), np.uint8)
            mask_array = cv2.dilate(edges, kernel, iterations=3)
            mask_array = (mask_array > 0).astype(np.uint8) * 255
            
            mask_ratio = mask_array.sum() / (width * height * 255)
            if 0.05 < mask_ratio < 0.95:
                mask_pil = Image.fromarray(mask_array, mode="L")
                return mask_pil
        except Exception as e:
            logger.warning(f"Edge-based detection failed: {e}")
        
        # If all methods fail, raise exception to trigger fallback
        raise RuntimeError("All salient detection methods failed")

    def _generate_mask_center(self, image: Image.Image) -> Image.Image:
        """
        Final fallback: generate a center region mask.
        """
        width, height = image.size
        mask = Image.new("L", (width, height), 0)  # Black (keep)
        
        # Create a white circle/ellipse in the center (edit region)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        
        # Center region (about 60% of image)
        center_x, center_y = width // 2, height // 2
        radius_x, radius_y = int(width * 0.3), int(height * 0.3)
        
        bbox = [
            center_x - radius_x, center_y - radius_y,
            center_x + radius_x, center_y + radius_y
        ]
        draw.ellipse(bbox, fill=255)  # White (edit)
        
        return mask

    def edit_image_auto(
        self,
        image_bytes: bytes,
        edit_prompt: str,
        strength: float,
        return_mask: bool = False
    ) -> Tuple[bytes, Optional[bytes]]:
        """
        Edit image automatically with AI-determined mask.
        
        Args:
            image_bytes: Source image as bytes
            edit_prompt: Text describing the edit
            strength: Edit strength (0.6-0.9 recommended)
            return_mask: Whether to return the generated mask
        
        Returns:
            Tuple of (edited_image_bytes, optional_mask_bytes)
        """
        # Load image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")
        
        # Validate image size (max 2048x2048)
        width, height = image.size
        if width > 2048 or height > 2048:
            # Resize if too large
            max_dim = max(width, height)
            scale = 2048 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Step A: Auto mask generation
        target_phrase = self._extract_target_object(edit_prompt)
        logger.info(f"Extracted target phrase: '{target_phrase}' from prompt: '{edit_prompt}'")
        
        if target_phrase is None:
            logger.warning("Could not extract target from prompt, using fallback mask")
        
        mask = self._generate_mask_segmentation(image, target_phrase)
        
        # Ensure mask matches image size exactly
        mask = mask.resize(image.size, Image.LANCZOS)
        
        # Step B: Inpainting
        logger.info(f"Running inpainting with strength={strength}")
        edited_image = self.editing_service.edit_image(
            image=image,
            prompt=edit_prompt,
            mask=mask,
            strength=strength,
        )
        
        # Convert to bytes
        edited_bytes_io = io.BytesIO()
        edited_image.save(edited_bytes_io, format="PNG")
        edited_bytes = edited_bytes_io.getvalue()
        
        # Convert mask to bytes if requested
        mask_bytes = None
        if return_mask:
            mask_bytes_io = io.BytesIO()
            mask.save(mask_bytes_io, format="PNG")
            mask_bytes = mask_bytes_io.getvalue()
        
        return edited_bytes, mask_bytes

