"""
Test script for the AI Planner
Tests 10 different prompts to ensure plan JSON is always valid and consistent.
"""
import json
from pathlib import Path
from services.planner import Planner
from schemas.plan import Plan


def test_planner():
    """Test the planner with 10 different prompts."""
    planner = Planner()
    
    test_cases = [
        {
            "prompt": "make background beach sunset and add text 'vacation mode'",
            "description": "Complex prompt with background change and text overlay"
        },
        {
            "prompt": "generate a cat wearing sunglasses",
            "description": "Simple image generation"
        },
        {
            "prompt": "change shirt to black",
            "description": "Simple edit with target extraction"
        },
        {
            "prompt": "create a video of a sunset",
            "description": "Video generation"
        },
        {
            "prompt": "edit the sky to make it more dramatic",
            "description": "Edit with target (sky)"
        },
        {
            "prompt": "add text 'Hello World' at the top",
            "description": "Text overlay with position"
        },
        {
            "prompt": "replace the background with a forest",
            "description": "Background replacement"
        },
        {
            "prompt": "make it photorealistic and upscale to 4k",
            "description": "Style and upscale operations"
        },
        {
            "prompt": "change the logo to red and add text 'NEW'",
            "description": "Multiple operations"
        },
        {
            "prompt": "generate a portrait of a person",
            "description": "Image generation with size hint"
        }
    ]
    
    print("=" * 80)
    print("AI PLANNER TEST SUITE")
    print("=" * 80)
    print()
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case["prompt"]
        description = test_case["description"]
        
        print(f"Test {i}/10: {description}")
        print(f"Prompt: '{prompt}'")
        print("-" * 80)
        
        try:
            # Create plan
            plan = planner.plan(prompt=prompt)
            
            # Validate plan structure
            assert isinstance(plan, Plan), "Plan must be a Plan object"
            assert plan.task_type in ["image_generate", "image_edit", "video_generate", "video_edit"], \
                f"Invalid task_type: {plan.task_type}"
            assert isinstance(plan.ops, list), "ops must be a list"
            assert isinstance(plan.params, dict), "params must be a dict"
            assert isinstance(plan.input_files, dict), "input_files must be a dict"
            
            # Check required params
            required_params = ["size", "strength", "steps", "duration", "fps", "guidance_scale"]
            for param in required_params:
                assert param in plan.params, f"Missing required param: {param}"
            
            # Validate operations
            for op in plan.ops:
                assert op.op_type in [
                    "inpaint", "outpaint", "overlay_text", "upscale",
                    "generate", "edit", "filter", "transform"
                ], f"Invalid op_type: {op.op_type}"
                assert isinstance(op.params, dict), "op.params must be a dict"
            
            # Convert to dict and JSON to ensure serialization works
            plan_dict = plan.to_dict()
            plan_json = plan.to_json()
            
            # Verify JSON is valid
            parsed = json.loads(plan_json)
            assert parsed["task_type"] == plan.task_type
            assert len(parsed["ops"]) == len(plan.ops)
            
            print("[PASSED]")
            print(f"  Task Type: {plan.task_type}")
            print(f"  Operations: {len(plan.ops)}")
            if plan.ops:
                for op in plan.ops:
                    print(f"    - {op.op_type}" + (f" (target: {op.target})" if op.target else ""))
            print(f"  Reasoning: {plan.reasoning[:100]}...")
            print()
            
        except AssertionError as e:
            print(f"[FAILED] {e}")
            print()
            all_passed = False
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            print()
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[FAILURE] SOME TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_planner()
    exit(0 if success else 1)

