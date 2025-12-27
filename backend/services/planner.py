"""
Planner Service
Analyzes user prompts and optional inputs to create execution plans.
"""
import json
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image
import re


class Planner:
    """
    Plans what actions to take based on user input.
    Analyzes prompts and determines the appropriate pipeline to execute.
    """
    
    def __init__(self):
        self.action_keywords = {
            "generate_image": [
                "generate", "create", "make", "draw", "render", "show me",
                "give me", "produce", "build", "design"
            ],
            "edit_image": [
                "edit", "modify", "change", "alter", "update", "transform",
                "adjust", "enhance", "improve", "fix", "replace", "add",
                "remove", "make it", "turn into", "convert to"
            ],
            "generate_video": [
                "video", "animate", "motion", "move", "cinematic", "film",
                "clip", "sequence", "animation"
            ]
        }
    
    def plan(
        self,
        prompt: str,
        input_file: Optional[Path] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an execution plan based on user input.
        
        Args:
            prompt: User's text prompt
            input_file: Optional input image/video file
            additional_params: Additional parameters (size, duration, etc.)
        
        Returns:
            JSON plan dict with:
            - action_type: "generate_image" | "edit_image" | "generate_video"
            - parameters: Action-specific parameters
            - input_files: Paths to input files
            - reasoning: Why this plan was chosen
        """
        prompt_lower = prompt.lower() if prompt else ""
        has_input_file = input_file is not None and input_file.exists()
        
        # Determine action type
        action_type = self._determine_action(prompt_lower, has_input_file)
        
        # Extract parameters from prompt and additional params
        parameters = self._extract_parameters(prompt, additional_params or {})
        
        # Build plan
        plan = {
            "action_type": action_type,
            "parameters": parameters,
            "input_files": {
                "input": str(input_file) if input_file else None
            },
            "reasoning": self._generate_reasoning(action_type, prompt, has_input_file)
        }
        
        return plan
    
    def _determine_action(self, prompt: str, has_input_file: bool) -> str:
        """Determine what action to take based on prompt and inputs."""
        
        # If user explicitly mentions video, prioritize that
        if any(keyword in prompt for keyword in self.action_keywords["generate_video"]):
            return "generate_video"
        
        # If user has input file and prompt suggests editing
        if has_input_file:
            if any(keyword in prompt for keyword in self.action_keywords["edit_image"]):
                return "edit_image"
            # If input file exists but prompt is about generation, still edit
            # (user probably wants to modify the uploaded image)
            return "edit_image"
        
        # Default: generate image from prompt
        return "generate_image"
    
    def _extract_parameters(self, prompt: str, additional_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from prompt and merge with additional params."""
        params = {
            "prompt": prompt,
            "size": additional_params.get("size", "512x512"),
            "style": additional_params.get("style"),
            "seed": additional_params.get("seed"),
            "strength": additional_params.get("strength", 0.75),
            "duration": additional_params.get("duration", 5),
            "overlay_text": additional_params.get("overlay_text"),
            "mask_prompt": None  # For automatic mask generation
        }
        
        # Extract style from prompt
        styles = ["photorealistic", "anime", "oil painting", "watercolor", 
                  "sketch", "3d render", "cartoon", "cyberpunk", "realistic"]
        for style in styles:
            if style in prompt.lower():
                params["style"] = style
                break
        
        # Extract size hints from prompt
        size_patterns = {
            "square": "512x512",
            "portrait": "512x768",
            "landscape": "768x512",
            "wide": "1024x512",
            "tall": "512x1024"
        }
        for keyword, size in size_patterns.items():
            if keyword in prompt.lower():
                params["size"] = size
                break
        
        # Extract strength for editing
        if "subtle" in prompt.lower() or "slight" in prompt.lower():
            params["strength"] = 0.4
        elif "strong" in prompt.lower() or "dramatic" in prompt.lower():
            params["strength"] = 0.9
        
        # Extract mask prompt (what to edit)
        # Look for phrases like "change the sky", "edit the background", etc.
        mask_keywords = ["sky", "background", "foreground", "face", "person", 
                        "object", "building", "car", "tree", "water"]
        for keyword in mask_keywords:
            if f"change the {keyword}" in prompt.lower() or \
               f"edit the {keyword}" in prompt.lower() or \
               f"modify the {keyword}" in prompt.lower():
                params["mask_prompt"] = keyword
                break
        
        return params
    
    def _generate_reasoning(self, action_type: str, prompt: str, has_input_file: bool) -> str:
        """Generate human-readable reasoning for the plan."""
        if action_type == "generate_image":
            return f"User wants to generate a new image from prompt: '{prompt[:50]}...'"
        elif action_type == "edit_image":
            if has_input_file:
                return f"User provided an image and wants to edit it based on: '{prompt[:50]}...'"
            else:
                return f"User wants to edit an image (but no input provided - will need input)"
        else:  # generate_video
            return f"User wants to generate a video from prompt: '{prompt[:50]}...'"
    
    def plan_to_json(self, plan: Dict[str, Any]) -> str:
        """Convert plan to JSON string."""
        return json.dumps(plan, indent=2)

