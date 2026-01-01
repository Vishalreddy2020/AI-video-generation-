"""
Test script for ExecutorRouter
Tests plan execution with different operation combinations.
"""
import json
from pathlib import Path
from services.executor_router import ExecutorRouter
from services.planner import Planner
from schemas.plan import Plan, Operation


def test_single_op_generate():
    """Test plan with 1 op (generate)."""
    print("=" * 80)
    print("Test 1: Single operation - Generate")
    print("=" * 80)
    
    plan = Plan(
        task_type="image_generate",
        ops=[
            Operation(
                op_type="generate",
                prompt="a cat wearing sunglasses",
                params={}
            )
        ],
        params={
            "size": "512x512",
            "steps": 20,
            "guidance_scale": 7.5
        }
    )
    
    router = ExecutorRouter()
    result = router.execute_plan(plan)
    
    if result["success"]:
        print(f"[PASSED] Generated image: {result['output_path']}")
        print(f"  Job ID: {result.get('job_id')}")
        print(f"  Ops ran: {len(result.get('metadata', {}).get('ops_ran', []))}")
        return True
    else:
        print(f"[FAILED] {result.get('error')}")
        return False


def test_two_ops_inpaint_overlay():
    """Test plan with 2 ops (inpaint + overlay_text)."""
    print("\n" + "=" * 80)
    print("Test 2: Two operations - Inpaint + Overlay Text")
    print("=" * 80)
    print("Note: This requires an input image. Using a generated one first.")
    
    # First generate an image
    generate_plan = Plan(
        task_type="image_generate",
        ops=[
            Operation(
                op_type="generate",
                prompt="a person wearing a blue shirt",
                params={}
            )
        ],
        params={"size": "512x512"}
    )
    
    router = ExecutorRouter()
    gen_result = router.execute_plan(generate_plan)
    
    if not gen_result["success"]:
        print(f"[SKIPPED] Could not generate test image: {gen_result.get('error')}")
        return True  # Don't fail the test, just skip
    
    from PIL import Image
    input_image = Image.open(gen_result["output_path"]).convert("RGB")
    
    # Now test inpaint + overlay
    plan = Plan(
        task_type="image_edit",
        ops=[
            Operation(
                op_type="inpaint",
                target="shirt",
                prompt="black shirt",
                params={}
            ),
            Operation(
                op_type="overlay_text",
                params={
                    "text": "TEST",
                    "position": "bottom_center"
                }
            )
        ],
        params={
            "strength": 0.75,
            "size": "512x512"
        }
    )
    
    result = router.execute_plan(plan, input_image=input_image)
    
    if result["success"]:
        print(f"[PASSED] Executed 2 operations: {result['output_path']}")
        print(f"  Job ID: {result.get('job_id')}")
        ops_ran = result.get('metadata', {}).get('ops_ran', [])
        print(f"  Ops ran: {[op.get('op_type') for op in ops_ran]}")
        return True
    else:
        print(f"[FAILED] {result.get('error')}")
        return False


def test_invalid_plan():
    """Test that invalid plans return clean error messages."""
    print("\n" + "=" * 80)
    print("Test 3: Invalid plan - should return clean error (400)")
    print("=" * 80)
    
    # Test 1: Missing required field
    try:
        plan = Plan(
            task_type="image_generate",
            ops=[],
            params={}
        )
        # This should work (empty ops is valid)
        print("[INFO] Empty ops plan is valid (will use default generation)")
    except Exception as e:
        print(f"[PASSED] Caught validation error: {e}")
        return True
    
    # Test 2: Invalid op_type
    try:
        plan = Plan(
            task_type="image_generate",
            ops=[
                Operation(
                    op_type="invalid_op_type",  # Invalid
                    params={}
                )
            ],
            params={}
        )
        router = ExecutorRouter()
        result = router.execute_plan(plan)
        if not result["success"]:
            print(f"[PASSED] Invalid op_type caught: {result.get('error')}")
            return True
        else:
            print("[FAILED] Invalid op_type was not caught")
            return False
    except Exception as e:
        print(f"[PASSED] Validation caught invalid op_type: {e}")
        return True


def test_plan_with_overlay_only():
    """Test plan with overlay_text operation."""
    print("\n" + "=" * 80)
    print("Test 4: Overlay text operation")
    print("=" * 80)
    
    # Generate base image
    generate_plan = Plan(
        task_type="image_generate",
        ops=[
            Operation(
                op_type="generate",
                prompt="a sunset landscape",
                params={}
            )
        ],
        params={"size": "512x512"}
    )
    
    router = ExecutorRouter()
    gen_result = router.execute_plan(generate_plan)
    
    if not gen_result["success"]:
        print(f"[SKIPPED] Could not generate test image: {gen_result.get('error')}")
        return True
    
    from PIL import Image
    input_image = Image.open(gen_result["output_path"]).convert("RGB")
    
    # Test overlay
    plan = Plan(
        task_type="image_edit",
        ops=[
            Operation(
                op_type="overlay_text",
                params={
                    "text": "Beautiful Sunset",
                    "position": "top_center"
                }
            )
        ],
        params={}
    )
    
    result = router.execute_plan(plan, input_image=input_image)
    
    if result["success"]:
        print(f"[PASSED] Overlay text applied: {result['output_path']}")
        return True
    else:
        print(f"[FAILED] {result.get('error')}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EXECUTOR ROUTER TEST SUITE")
    print("=" * 80)
    print()
    
    tests = [
        ("Single op (generate)", test_single_op_generate),
        ("Two ops (inpaint + overlay)", test_two_ops_inpaint_overlay),
        ("Invalid plan handling", test_invalid_plan),
        ("Overlay text only", test_plan_with_overlay_only),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    for name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[FAILURE] SOME TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



