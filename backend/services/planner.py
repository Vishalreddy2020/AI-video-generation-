"""
Planner Service
Analyzes user prompts and converts them into structured execution plans.

Supports two modes:
- v1 (rule-based): Fast, deterministic, no API costs, works offline
- v2 (LLM-based): More intelligent, understands context, requires LLM access
"""
import re
import json
import os
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path

from schemas.plan import Plan, Operation


class Planner:
    """
    Planner that converts high-level prompts into execution plans.
    
    Can operate in two modes:
    - "rule" (default): Fast rule-based pattern matching
    - "llm": Uses LLM for intelligent understanding (requires LLM access)
    """
    
    def __init__(self, mode: Literal["rule", "llm", "auto"] = "auto"):
        """
        Initialize planner.
        
        Args:
            mode: "rule" (fast, hardcoded patterns), "llm" (AI-powered), 
                  or "auto" (try LLM, fallback to rule)
        """
        self.mode = mode
        self.llm_available = False
        self.llm_type = None
        self._init_llm()
        # Keywords for detecting task types
        self.task_keywords = {
            "image_generate": [
                "generate", "create", "make", "draw", "render", "show me",
                "give me", "produce", "build", "design", "new image"
            ],
            "image_edit": [
                "edit", "modify", "change", "alter", "update", "transform",
                "adjust", "enhance", "improve", "fix", "replace", "add",
                "remove", "make it", "turn into", "convert to"
            ],
            "video_generate": [
                "video", "animate", "motion", "move", "cinematic", "film",
                "clip", "sequence", "animation", "make video"
            ],
            "video_edit": [
                "edit video", "modify video", "change video", "video edit"
            ]
        }
        
        # Keywords for detecting operations
        self.operation_patterns = {
            "overlay_text": [
                r"add text ['\"]([^'\"]+)['\"]",
                r"add text (.+?)(?: and|,|$)",
                r"text ['\"]([^'\"]+)['\"]",
                r"overlay ['\"]([^'\"]+)['\"]",
                r"write ['\"]([^'\"]+)['\"]",
                r"put ['\"]([^'\"]+)['\"]",
            ],
            "inpaint": [
                r"change (?:the )?(background|sky|foreground|shirt|pants|dress|face|hair|eyes)",
                r"replace (?:the )?(background|sky|foreground|shirt|pants|dress|face|hair|eyes)",
                r"edit (?:the )?(background|sky|foreground|shirt|pants|dress|face|hair|eyes)",
                r"modify (?:the )?(background|sky|foreground|shirt|pants|dress|face|hair|eyes)",
                r"make (?:the )?(background|sky|foreground|shirt|pants|dress|face|hair|eyes)",
            ],
            "outpaint": [
                r"extend (?:the )?(background|image|canvas)",
                r"expand (?:the )?(background|image|canvas)",
                r"add (?:more )?(background|space|area)",
            ],
            "upscale": [
                r"upscale", r"enlarge", r"make (?:it )?bigger", r"increase (?:the )?size",
                r"high resolution", r"hd", r"4k"
            ]
        }
    
    def _init_llm(self):
        """Initialize LLM if available (for v2 mode)."""
        if self.mode in ["llm", "auto"]:
            # Try to load a local LLM (Ollama, transformers, etc.)
            try:
                # Option 1: Try Ollama (local, free)
                import requests
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        self.llm_available = True
                        self.llm_type = "ollama"
                        print("✓ LLM available (Ollama)")
                        return
                except:
                    pass
                
                # Option 2: Try OpenAI API (if key provided)
                if os.getenv("OPENAI_API_KEY"):
                    self.llm_available = True
                    self.llm_type = "openai"
                    print("✓ LLM available (OpenAI)")
                    return
                
                # Option 3: Try local transformers model (if available)
                try:
                    from transformers import pipeline
                    # Don't load immediately, just check if available
                    self.llm_available = True
                    self.llm_type = "transformers"
                    print("✓ LLM available (Transformers)")
                    return
                except:
                    pass
                    
            except Exception as e:
                print(f"LLM initialization check failed: {e}")
        
        if self.mode == "llm" and not self.llm_available:
            print("⚠ Warning: LLM mode requested but no LLM available. Falling back to rule-based.")
        elif self.mode == "auto":
            print("ℹ Using rule-based planner (LLM not available)")
    
    def plan(
        self,
        prompt: str,
        input_file: Optional[Path] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        media_type: Optional[str] = None,
        force_mode: Optional[Literal["rule", "llm"]] = None
    ) -> Plan:
        """
        Create an execution plan from user prompt.
        
        Args:
            prompt: User's high-level prompt
            input_file: Optional input image/video file
            additional_params: Additional parameters (size, duration, etc.)
            media_type: Optional hint ("auto", "image", "video")
            force_mode: Override planner mode for this call ("rule" or "llm")
        
        Returns:
            Plan object with task_type, ops, params, and input_files
        """
        # Determine which mode to use
        use_llm = False
        if force_mode == "llm":
            use_llm = True
        elif force_mode == "rule":
            use_llm = False
        elif self.mode == "llm" and self.llm_available:
            use_llm = True
        elif self.mode == "auto" and self.llm_available:
            # Auto mode: use LLM if available
            use_llm = True
        
        # Use LLM-based planning if available and requested
        if use_llm:
            try:
                return self._plan_with_llm(prompt, input_file, additional_params, media_type)
            except Exception as e:
                print(f"LLM planning failed: {e}, falling back to rule-based")
                # Fall through to rule-based
        
        # Rule-based planning (v1)
        return self._plan_with_rules(prompt, input_file, additional_params, media_type)
    
    def _plan_with_llm(
        self,
        prompt: str,
        input_file: Optional[Path],
        additional_params: Optional[Dict[str, Any]],
        media_type: Optional[str]
    ) -> Plan:
        """Plan using LLM (v2 - intelligent understanding)."""
        # Get LLM response
        llm_response = self._query_llm(prompt, input_file, media_type)
        
        # Parse LLM response into Plan structure
        return self._parse_llm_response(llm_response, prompt, input_file, additional_params)
    
    def _query_llm(self, prompt: str, input_file: Optional[Path], media_type: Optional[str]) -> str:
        """Query LLM for plan generation."""
        system_prompt = """You are an AI planning assistant. Convert user prompts into structured execution plans.

Output a JSON plan with this structure:
{
  "task_type": "image_generate" | "image_edit" | "video_generate" | "video_edit",
  "ops": [
    {
      "op_type": "inpaint" | "outpaint" | "overlay_text" | "upscale" | "generate" | "edit",
      "target": "background" | "sky" | "shirt" | etc. (optional),
      "prompt": "operation-specific prompt",
      "params": {}
    }
  ],
  "params": {
    "size": "512x512",
    "strength": 0.75,
    "steps": 20,
    "duration": 5,
    "fps": 24
  }
}

Examples:
- "make background beach sunset and add text 'vacation mode'" → 
  {"task_type": "image_edit", "ops": [{"op_type": "inpaint", "target": "background", "prompt": "beach sunset"}, {"op_type": "overlay_text", "params": {"text": "vacation mode", "position": "bottom"}}]}
- "generate a cat wearing sunglasses" → 
  {"task_type": "image_generate", "ops": [{"op_type": "generate", "prompt": "a cat wearing sunglasses"}]}
"""
        
        user_prompt = f"User prompt: {prompt}"
        if input_file:
            user_prompt += f"\nInput file: {input_file.name}"
        if media_type:
            user_prompt += f"\nMedia type hint: {media_type}"
        
        if self.llm_type == "ollama":
            return self._query_ollama(system_prompt, user_prompt)
        elif self.llm_type == "openai":
            return self._query_openai(system_prompt, user_prompt)
        elif self.llm_type == "transformers":
            return self._query_transformers(system_prompt, user_prompt)
        else:
            raise RuntimeError("LLM type not properly initialized")
    
    def _query_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Query Ollama (local LLM)."""
        import requests
        
        # Use a small, fast model like llama3.2 or mistral
        model = "llama3.2"  # or "mistral", "phi3", etc.
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\n{user_prompt}\n\nOutput JSON plan:",
                "stream": False,
                "format": "json"  # Request JSON output
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def _query_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Query OpenAI API."""
        import openai
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nOutput JSON plan:"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _query_transformers(self, system_prompt: str, user_prompt: str) -> str:
        """Query local transformers model."""
        from transformers import pipeline
        
        # Use a small instruction-following model
        if not hasattr(self, '_transformers_pipe'):
            self._transformers_pipe = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct",  # Small, fast
                device_map="auto"
            )
        
        prompt = f"{system_prompt}\n\n{user_prompt}\n\nOutput JSON plan:"
        response = self._transformers_pipe(
            prompt,
            max_new_tokens=500,
            return_full_text=False,
            temperature=0.3
        )
        return response[0]["generated_text"]
    
    def _parse_llm_response(
        self,
        llm_response: str,
        prompt: str,
        input_file: Optional[Path],
        additional_params: Optional[Dict[str, Any]]
    ) -> Plan:
        """Parse LLM JSON response into Plan object."""
        try:
            # Extract JSON from response (might have markdown code blocks)
            json_str = llm_response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            plan_dict = json.loads(json_str)
            
            # Convert to Plan object
            ops = [Operation(**op) for op in plan_dict.get("ops", [])]
            
            # Merge with additional params
            params = plan_dict.get("params", {})
            if additional_params:
                params.update(additional_params)
            
            return Plan(
                task_type=plan_dict["task_type"],
                ops=ops,
                params=params,
                input_files={"input": str(input_file) if input_file else None},
                reasoning=f"LLM-generated plan. Original prompt: '{prompt[:100]}'"
            )
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {llm_response}")
    
    def _plan_with_rules(
        self,
        prompt: str,
        input_file: Optional[Path],
        additional_params: Optional[Dict[str, Any]],
        media_type: Optional[str]
    ) -> Plan:
        """Rule-based planning (v1 - fast, deterministic)."""
        prompt_lower = prompt.lower() if prompt else ""
        has_input_file = input_file is not None and input_file.exists()
        additional_params = additional_params or {}
        
        # Determine task type
        task_type = self._determine_task_type(prompt_lower, has_input_file, media_type)
        
        # Extract operations from prompt
        ops = self._extract_operations(prompt, prompt_lower)
        
        # Extract global parameters
        params = self._extract_params(prompt, prompt_lower, additional_params)
        
        # Build input files dict
        input_files = {
            "input": str(input_file) if input_file else None
        }
        
        # Generate reasoning
        reasoning = self._generate_reasoning(task_type, prompt, has_input_file, ops)
        
        # Create and validate plan
        plan = Plan(
            task_type=task_type,
            ops=ops,
            params=params,
            input_files=input_files,
            reasoning=reasoning
        )
        
        return plan
    
    def _determine_task_type(
        self, 
        prompt_lower: str, 
        has_input_file: bool,
        media_type: Optional[str]
    ) -> str:
        """Determine the task type based on prompt and context."""
        
        # If media_type is explicitly set, use it
        if media_type and media_type != "auto":
            if media_type == "image":
                return "image_edit" if has_input_file else "image_generate"
            elif media_type == "video":
                return "video_edit" if has_input_file else "video_generate"
        
        # Check for video keywords
        if any(keyword in prompt_lower for keyword in self.task_keywords["video_edit"]):
            return "video_edit"
        if any(keyword in prompt_lower for keyword in self.task_keywords["video_generate"]):
            return "video_generate"
        
        # If input file exists, it's likely an edit task
        if has_input_file:
            if any(keyword in prompt_lower for keyword in self.task_keywords["image_edit"]):
                return "image_edit"
            # Default to edit if file provided
            return "image_edit"
        
        # Check for image generation keywords
        if any(keyword in prompt_lower for keyword in self.task_keywords["image_generate"]):
            return "image_generate"
        
        # Default: generate image
        return "image_generate"
    
    def _extract_operations(self, prompt: str, prompt_lower: str) -> List[Operation]:
        """Extract operations from the prompt."""
        ops = []
        
        # Detect overlay_text operation
        text_content = self._extract_text_overlay(prompt, prompt_lower)
        if text_content:
            position = self._extract_text_position(prompt_lower)
            ops.append(Operation(
                op_type="overlay_text",
                prompt=text_content,
                params={
                    "text": text_content,
                    "position": position,
                    "font_size": None  # Auto-calculated
                }
            ))
        
        # Detect inpaint operation
        inpaint_target = self._extract_inpaint_target(prompt, prompt_lower)
        if inpaint_target:
            inpaint_prompt = self._extract_inpaint_prompt(prompt, inpaint_target)
            ops.append(Operation(
                op_type="inpaint",
                target=inpaint_target,
                prompt=inpaint_prompt,
                params={}
            ))
        
        # Detect outpaint operation
        if self._has_outpaint_keywords(prompt_lower):
            ops.append(Operation(
                op_type="outpaint",
                params={}
            ))
        
        # Detect upscale operation
        if self._has_upscale_keywords(prompt_lower):
            ops.append(Operation(
                op_type="upscale",
                params={"scale": 2}  # Default 2x upscale
            ))
        
        # If no operations detected but it's an edit task, add a generic edit op
        if not ops and any(kw in prompt_lower for kw in self.task_keywords["image_edit"]):
            ops.append(Operation(
                op_type="edit",
                prompt=prompt,
                params={}
            ))
        
        return ops
    
    def _extract_text_overlay(self, prompt: str, prompt_lower: str) -> Optional[str]:
        """Extract text to overlay from prompt."""
        # Try patterns in order
        for pattern in self.operation_patterns["overlay_text"]:
            match = re.search(pattern, prompt_lower, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Remove quotes if present
                text = text.strip('"\'')
                if text:
                    return text
        
        # Fallback: look for "add text" followed by quoted text
        if "add text" in prompt_lower or "text" in prompt_lower:
            # Try to find quoted text
            quoted_match = re.search(r'["\']([^"\']+)["\']', prompt)
            if quoted_match:
                return quoted_match.group(1)
        
        return None
    
    def _extract_text_position(self, prompt_lower: str) -> str:
        """Extract text position from prompt."""
        position_keywords = {
            "top": ["top"],
            "bottom": ["bottom"],
            "center": ["center", "middle"],
            "top_left": ["top left", "upper left"],
            "top_right": ["top right", "upper right"],
            "bottom_left": ["bottom left", "lower left"],
            "bottom_right": ["bottom right", "lower right"]
        }
        
        for position, keywords in position_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                return position
        
        return "bottom_center"  # Default
    
    def _extract_inpaint_target(self, prompt: str, prompt_lower: str) -> Optional[str]:
        """Extract target region for inpainting."""
        for pattern in self.operation_patterns["inpaint"]:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1)
        
        # Fallback: look for common targets
        common_targets = [
            "background", "sky", "foreground", "shirt", "pants", "dress",
            "face", "hair", "eyes", "person", "logo", "text", "sign"
        ]
        for target in common_targets:
            if target in prompt_lower:
                return target
        
        return None
    
    def _extract_inpaint_prompt(self, prompt: str, target: Optional[str]) -> str:
        """Extract the inpainting prompt (what to change the target to)."""
        # Try to extract what the target should become
        # Example: "change background to beach" -> "beach"
        patterns = [
            rf"change (?:the )?{target} to (.+?)(?: and|,|$)",
            rf"replace (?:the )?{target} with (.+?)(?: and|,|$)",
            rf"make (?:the )?{target} (.+?)(?: and|,|$)",
            rf"{target} should be (.+?)(?: and|,|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                return match.group(1).strip()
        
        # If no specific target prompt, use the full prompt
        return prompt
    
    def _has_outpaint_keywords(self, prompt_lower: str) -> bool:
        """Check if prompt contains outpaint keywords."""
        return any(
            re.search(pattern, prompt_lower)
            for pattern in self.operation_patterns["outpaint"]
        )
    
    def _has_upscale_keywords(self, prompt_lower: str) -> bool:
        """Check if prompt contains upscale keywords."""
        return any(
            re.search(pattern, prompt_lower)
            for pattern in self.operation_patterns["upscale"]
        )
    
    def _extract_params(
        self, 
        prompt: str, 
        prompt_lower: str,
        additional_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract global parameters from prompt and additional params."""
        params = {
            "size": additional_params.get("size", "512x512"),
            "strength": additional_params.get("strength", 0.75),
            "steps": additional_params.get("steps", 20),
            "seed": additional_params.get("seed"),
            "duration": additional_params.get("duration", 5),
            "fps": additional_params.get("fps", 24),
            "guidance_scale": additional_params.get("guidance_scale", 7.5)
        }
        
        # Extract style from prompt
        styles = [
            "photorealistic", "anime", "oil painting", "watercolor",
            "sketch", "3d render", "cartoon", "cyberpunk", "realistic"
        ]
        for style in styles:
            if style in prompt_lower:
                params["style"] = style
                break
        
        # Extract size hints
        size_patterns = {
            "square": "512x512",
            "portrait": "512x768",
            "landscape": "768x512",
            "wide": "1024x512",
            "tall": "512x1024"
        }
        for keyword, size in size_patterns.items():
            if keyword in prompt_lower:
                params["size"] = size
                break
        
        # Extract strength
        if "subtle" in prompt_lower or "slight" in prompt_lower:
            params["strength"] = 0.4
        elif "strong" in prompt_lower or "dramatic" in prompt_lower:
            params["strength"] = 0.9
        
        return params
    
    def _generate_reasoning(
        self,
        task_type: str,
        prompt: str,
        has_input_file: bool,
        ops: List[Operation]
    ) -> str:
        """Generate human-readable reasoning for the plan."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Task type: {task_type}")
        
        if has_input_file:
            reasoning_parts.append("Input file provided")
        else:
            reasoning_parts.append("No input file (generation task)")
        
        if ops:
            op_descriptions = [f"{op.op_type}" + (f" (target: {op.target})" if op.target else "") for op in ops]
            reasoning_parts.append(f"Operations: {', '.join(op_descriptions)}")
        else:
            reasoning_parts.append("No specific operations detected (using default)")
        
        return ". ".join(reasoning_parts) + f". Original prompt: '{prompt[:100]}'"
    
    def plan_to_dict(self, plan: Plan) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return plan.to_dict()
    
    def plan_to_json(self, plan: Plan) -> str:
        """Convert plan to JSON string."""
        return plan.to_json()
