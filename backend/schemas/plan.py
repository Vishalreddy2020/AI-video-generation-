"""
Plan Schema
Defines the structure and validation for execution plans.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator


class Operation(BaseModel):
    """A single operation in the plan."""
    op_type: Literal[
        "inpaint",
        "outpaint",
        "overlay_text",
        "upscale",
        "generate",
        "edit",
        "filter",
        "transform"
    ] = Field(..., description="Type of operation")
    target: Optional[str] = Field(None, description="Target region/object (e.g., 'background', 'shirt')")
    prompt: Optional[str] = Field(None, description="Operation-specific prompt")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operation-specific parameters")


class Plan(BaseModel):
    """Execution plan for AI operations."""
    task_type: Literal[
        "image_generate",
        "image_edit",
        "video_generate",
        "video_edit"
    ] = Field(..., description="Type of task")
    ops: List[Operation] = Field(default_factory=list, description="List of operations to perform")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global parameters (size, strength, steps, seed, duration, fps, etc.)"
    )
    input_files: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Input file paths (input, mask, face, etc.)"
    )
    reasoning: Optional[str] = Field(None, description="Human-readable reasoning for the plan")
    
    @validator('ops')
    def validate_ops_not_empty_for_edit(cls, v, values):
        """Ensure ops list is not empty for edit tasks."""
        task_type = values.get('task_type')
        if task_type in ['image_edit', 'video_edit'] and not v:
            raise ValueError(f"task_type '{task_type}' requires at least one operation")
        return v
    
    @validator('params')
    def validate_params(cls, v):
        """Ensure default params are set."""
        defaults = {
            "size": "512x512",
            "strength": 0.75,
            "steps": 20,
            "seed": None,
            "duration": 5,
            "fps": 24,
            "guidance_scale": 7.5
        }
        for key, default_value in defaults.items():
            if key not in v:
                v[key] = default_value
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convert plan to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

