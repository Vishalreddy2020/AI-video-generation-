from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import uvicorn
import os
import shutil
from pathlib import Path
from typing import Optional
import tempfile
import asyncio

from services.video_generator import VideoGenerator
from services.face_swapper import FaceSwapper
from services.planner import Planner
from services.executor import Executor
from services.executors.image_generate import ImageGenerateExecutor
from services.executors.image_edit_inpaint import ImageEditInpaintExecutor
from services.executors.image_edit_auto import ImageEditAutoExecutor
from services.executor_router import ExecutorRouter
from schemas.plan import Plan

app = FastAPI(title="AI Video Generator")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
video_generator = VideoGenerator()
face_swapper = FaceSwapper()
planner = Planner()
executor = Executor()
image_generate_executor = ImageGenerateExecutor()
image_edit_inpaint_executor = ImageEditInpaintExecutor()
image_edit_auto_executor = ImageEditAutoExecutor()
executor_router = ExecutorRouter()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {"message": "AI Video Generator API"}


@app.post("/api/plan")
async def create_plan(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None),
    size: Optional[str] = Form("512x512"),
    duration: Optional[int] = Form(5),
    strength: Optional[float] = Form(0.75),
    style: Optional[str] = Form(None),
    seed: Optional[int] = Form(None)
):
    """
    Create an execution plan based on user input.
    Returns JSON plan that can be executed.
    """
    try:
        # Save uploaded file if provided
        input_path = None
        if file:
            input_path = UPLOAD_DIR / f"input_{file.filename}"
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Create plan
        additional_params = {
            "size": size,
            "duration": duration,
            "strength": strength,
            "style": style,
            "seed": seed
        }
        
        plan = planner.plan(
            prompt=prompt,
            input_file=input_path,
            additional_params=additional_params
        )
        
        return JSONResponse(content=plan.to_dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/plan")
async def agent_plan(
    prompt: str = Form(...),
    media_type: Optional[str] = Form("auto"),
    file: Optional[UploadFile] = File(None),
    size: Optional[str] = Form(None),
    duration: Optional[int] = Form(None),
    strength: Optional[float] = Form(None),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    fps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None)
):
    """
    AI Planner endpoint - converts high-level prompts into structured execution plans.
    
    This is the "think" layer that analyzes user prompts and creates a JSON plan
    that executors can run.
    
    Inputs:
      - prompt: High-level user prompt (e.g., "make background beach sunset and add text 'vacation mode'")
      - media_type: "auto" | "image" | "video" (hint for task type)
      - file: Optional input image/video file
      - size, duration, strength, steps, seed, fps, guidance_scale: Optional parameters
    
    Output:
      - JSON plan with:
        - task_type: image_generate | image_edit | video_generate | video_edit
        - ops: List of operations (inpaint, overlay_text, upscale, etc.)
        - params: Global parameters (size, strength, steps, seed, duration, fps, etc.)
        - input_files: Input file paths
        - reasoning: Human-readable explanation
    """
    try:
        # Validate prompt
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt cannot be empty")
        
        # Validate media_type
        if media_type not in ["auto", "image", "video"]:
            raise HTTPException(
                status_code=400,
                detail="media_type must be 'auto', 'image', or 'video'"
            )
        
        # Save uploaded file if provided
        input_path = None
        if file:
            input_path = UPLOAD_DIR / f"input_{file.filename}"
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Build additional params (only include non-None values)
        additional_params = {}
        if size:
            additional_params["size"] = size
        if duration:
            additional_params["duration"] = duration
        if strength:
            additional_params["strength"] = strength
        if steps:
            additional_params["steps"] = steps
        if seed:
            additional_params["seed"] = seed
        if fps:
            additional_params["fps"] = fps
        if guidance_scale:
            additional_params["guidance_scale"] = guidance_scale
        
        # Create plan using the planner
        plan = planner.plan(
            prompt=prompt.strip(),
            input_file=input_path,
            additional_params=additional_params if additional_params else None,
            media_type=media_type
        )
        
        # Return plan as JSON
        return JSONResponse(content=plan.to_dict())
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


@app.post("/agent/execute")
async def agent_execute(
    plan_json: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    mask_file: Optional[UploadFile] = File(None)
):
    """
    Execute a plan end-to-end.
    
    This is the "smart" endpoint that takes a plan JSON and executes all operations
    in sequence, chaining outputs between operations.
    
    Inputs:
      - plan_json: JSON string of the plan (from /agent/plan)
      - image_file: Optional input image (if not in plan)
      - mask_file: Optional input mask (if not in plan)
    
    Output:
      - Final image/video file
      - Metadata (job_id, ops_ran, seed, etc.)
    """
    try:
        import json
        from PIL import Image
        import io
        
        # Parse plan JSON
        try:
            plan_dict = json.loads(plan_json)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid plan JSON: {str(e)}"
            )
        
        # Validate and create Plan object
        try:
            plan = Plan(**plan_dict)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid plan structure: {str(e)}"
            )
        
        # Load input image if provided
        input_image = None
        if image_file:
            image_bytes = await image_file.read()
            try:
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file: {str(e)}"
                )
        elif plan.input_files.get("input"):
            # Load from plan input_files
            input_path = Path(plan.input_files["input"])
            if input_path.exists():
                input_image = Image.open(input_path).convert("RGB")
        
        # Load input mask if provided
        input_mask = None
        if mask_file:
            mask_bytes = await mask_file.read()
            try:
                input_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid mask file: {str(e)}"
                )
        
        # Execute plan
        result = executor_router.execute_plan(
            plan=plan,
            input_image=input_image,
            input_mask=input_mask
        )
        
        if not result.get("success"):
            error_msg = result.get("error", "Execution failed")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        output_path = Path(result["output_path"])
        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Output file was not created"
            )
        
        # Return file with metadata in headers
        response = FileResponse(
            output_path,
            media_type="image/png",
            filename=output_path.name
        )
        
        # Add metadata to response headers
        metadata = result.get("metadata", {})
        response.headers["X-Job-ID"] = result.get("job_id", "")
        response.headers["X-Output-Type"] = result.get("output_type", "image")
        if metadata.get("seed"):
            response.headers["X-Seed"] = str(metadata["seed"])
        if metadata.get("ops_ran"):
            ops_str = ",".join([op.get("op_type", "") for op in metadata["ops_ran"]])
            response.headers["X-Ops-Ran"] = ops_str
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(e)}"
        )


@app.post("/api/execute")
async def execute_plan(
    plan: str = Form(...),  # JSON string of plan
    file: Optional[UploadFile] = File(None)  # Optional file if not in plan
):
    """
    Execute a plan created by /api/plan endpoint.
    """
    try:
        import json
        plan_dict = json.loads(plan)
        
        # If file provided and not in plan, add it
        if file and not plan_dict.get("input_files", {}).get("input"):
            input_path = UPLOAD_DIR / f"input_{file.filename}"
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            plan_dict.setdefault("input_files", {})["input"] = str(input_path)
        
        # Execute plan
        result = await executor.execute(plan_dict)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Execution failed"))
        
        output_path = Path(result["output_path"])
        output_type = result["output_type"]
        
        if output_type == "image":
            return FileResponse(
                output_path,
                media_type="image/png",
                filename=output_path.name
            )
        else:  # video
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename=output_path.name
            )
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid plan JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None),
    size: Optional[str] = Form("512x512"),
    duration: Optional[int] = Form(5),
    strength: Optional[float] = Form(0.75),
    style: Optional[str] = Form(None),
    seed: Optional[int] = Form(None)
):
    """
    Unified endpoint: Plans and executes in one call.
    This is the main endpoint for the UI to use.
    """
    try:
        # Save uploaded file if provided
        input_path = None
        if file:
            input_path = UPLOAD_DIR / f"input_{file.filename}"
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Create plan
        additional_params = {
            "size": size,
            "duration": duration,
            "strength": strength,
            "style": style,
            "seed": seed
        }
        
        plan = planner.plan(
            prompt=prompt,
            input_file=input_path,
            additional_params=additional_params
        )
        
        # Debug: Print plan structure
        print(f"Generated plan: task_type={plan.task_type}, ops_count={len(plan.ops)}")
        for i, op in enumerate(plan.ops):
            print(f"  Op {i+1}: {op.op_type}, prompt='{op.prompt[:50]}...'")
        
        # Load input image if file was provided
        input_image = None
        if input_path and input_path.exists():
            from PIL import Image
            try:
                input_image = Image.open(input_path).convert("RGB")
                print(f"Loaded input image: {input_image.size}")
            except Exception as e:
                print(f"Warning: Could not load input image: {e}")
        
        # Execute plan using executor_router (handles Plan objects correctly)
        # Note: executor_router.execute_plan is synchronous, not async
        import traceback
        try:
            print("Starting plan execution...")
            result = executor_router.execute_plan(
                plan=plan,
                input_image=input_image,
                input_mask=None
            )
            print(f"Execution result: success={result.get('success')}, output_path={result.get('output_path')}")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
        
        if not result.get("success"):
            error_msg = result.get("error", "Execution failed")
            print(f"Execution failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        output_path = Path(result["output_path"])
        output_type = result["output_type"]
        
        # Verify file exists
        if not output_path.exists():
            error_msg = f"Output file not found: {output_path}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"Sending file: {output_path} (size: {output_path.stat().st_size} bytes)")
        
        if output_type == "image":
            return FileResponse(
                output_path,
                media_type="image/png",
                filename=output_path.name
            )
        else:  # video
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename=output_path.name
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoint for backward compatibility
@app.post("/api/generate-video")
async def generate_video(
    file: Optional[UploadFile] = File(None),
    text_prompt: Optional[str] = Form(None),
    duration: int = Form(5),
    replace_face: bool = Form(False),
    face_image: Optional[UploadFile] = File(None)
):
    """
    Generate video from photo, text, or video input.
    Legacy endpoint - maintained for backward compatibility.
    """
    try:
        if duration not in [5, 10]:
            raise HTTPException(status_code=400, detail="Duration must be 5 or 10 seconds")
        
        # Save uploaded files
        input_path = None
        face_path = None
        
        if file:
            input_path = UPLOAD_DIR / f"input_{file.filename}"
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        if replace_face and face_image:
            face_path = UPLOAD_DIR / f"face_{face_image.filename}"
            with open(face_path, "wb") as buffer:
                shutil.copyfileobj(face_image.file, buffer)
        
        # Generate video
        output_path = await video_generator.generate(
            input_path=input_path,
            text_prompt=text_prompt,
            duration=duration
        )
        
        # Replace face if requested
        if replace_face and face_path and output_path:
            try:
                output_path = await face_swapper.swap_face(
                    source_video=output_path,
                    target_face=face_path
                )
            except ValueError as e:
                error_msg = str(e)
                if "face_recognition" in error_msg.lower():
                    raise HTTPException(
                        status_code=500,
                        detail="Face recognition library not installed. Run install_face_recognition.bat to install it. The app will use OpenCV face detection as fallback."
                    )
                raise HTTPException(status_code=500, detail=f"Face replacement error: {error_msg}")
        
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Video generation failed")
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"generated_video_{duration}s.mp4"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image/generate")
async def generate_image(
    prompt: str = Form(...),
    width: int = Form(512),
    height: int = Form(512),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    seed: Optional[int] = Form(None)
):
    """
    Generate an image from a text prompt using OpenVINO (optimized for Intel Arc).
    
    Args:
        prompt: Text description of the image to generate
        width: Image width (default: 512, must be multiple of 8)
        height: Image height (default: 512, must be multiple of 8)
        num_inference_steps: Number of denoising steps (default: 20)
        guidance_scale: Guidance scale (default: 7.5)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Generated image file (PNG)
    """
    try:
        # Validate dimensions
        if width < 256 or width > 1024 or height < 256 or height > 1024:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be between 256 and 1024"
            )
        
        # Generate image
        output_path = image_generate_executor.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Image generation failed")
        
        return FileResponse(
            output_path,
            media_type="image/png",
            filename=output_path.name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image/edit")
async def edit_image(
    image_file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = File(None),
    edit_prompt: str = Form(...),
    strength: float = Form(0.75),
):
    """
    High-level local image editing using mask-based inpainting.

    Inputs:
      - image_file: base image to edit
      - mask_file: optional mask (white = edit, black = keep)
      - edit_prompt: text describing the edit (e.g., "change shirt to black")
      - strength: 0.6–0.9 recommended for visible edits
    """
    try:
        # Validate strength (allow 0.0–1.0 but recommend 0.6–0.9)
        if not 0.0 <= strength <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="strength must be between 0.0 and 1.0",
            )

        # Load source image
        from PIL import Image
        import io

        image_bytes = await image_file.read()
        try:
            src_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image_file")

        # Load mask if provided
        mask_image = None
        if mask_file is not None:
            mask_bytes = await mask_file.read()
            try:
                # Convert to grayscale; white = edit, black = keep
                mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid mask_file")

        # Run inpainting executor
        output_path = image_edit_inpaint_executor.edit(
            image=src_image,
            mask=mask_image,
            prompt=edit_prompt,
            strength=strength,
        )

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Image editing failed")

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=output_path.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image/edit/auto")
async def edit_image_auto(
    image_file: UploadFile = File(...),
    edit_prompt: str = Form(...),
    strength: float = Form(0.75),
    return_mask: bool = Form(False),
):
    """
    Automatic image editing where the AI determines what to edit.
    
    The AI automatically generates a mask based on the edit prompt and then
    performs inpainting on the selected region.
    
    Inputs (multipart/form-data):
      - image_file: base image to edit
      - edit_prompt: text describing the edit (e.g., "change shirt to black")
      - strength: edit strength (0.6-0.9 recommended, default: 0.75)
      - return_mask: whether to return the generated mask (default: false)
    
    Output (JSON):
      - edited_image: base64-encoded PNG image
      - mask_image: base64-encoded PNG mask (only if return_mask=true)
    """
    try:
        import base64
        
        # Validate strength range
        if not 0.6 <= strength <= 0.9:
            raise HTTPException(
                status_code=400,
                detail="strength must be between 0.6 and 0.9"
            )
        
        # Validate edit_prompt
        if not edit_prompt or not edit_prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="edit_prompt cannot be empty"
            )
        
        # Read image file
        image_bytes = await image_file.read()
        
        # Validate image size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file too large (max 10MB)"
            )
        
        # Validate it's actually an image
        from PIL import Image
        import io
        try:
            test_image = Image.open(io.BytesIO(image_bytes))
            test_image.verify()
            # Get image dimensions for logging
            test_image = Image.open(io.BytesIO(image_bytes))  # Reopen after verify
            width, height = test_image.size
            if width > 2048 or height > 2048:
                # Log warning but don't fail - executor will resize
                pass
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # Run automatic editing
        edited_bytes, mask_bytes = image_edit_auto_executor.edit_image_auto(
            image_bytes=image_bytes,
            edit_prompt=edit_prompt.strip(),
            strength=strength,
            return_mask=return_mask
        )
        
        # Convert to base64
        edited_base64 = base64.b64encode(edited_bytes).decode('utf-8')
        
        response = {
            "edited_image_base64": edited_base64,
        }
        
        if return_mask and mask_bytes:
            mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
            response["mask_image_base64"] = mask_base64
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
