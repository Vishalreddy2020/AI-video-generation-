from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import os
import shutil
from pathlib import Path
from typing import Optional
import tempfile
import asyncio

from services.video_generator import VideoGenerator
from services.face_swapper import FaceSwapper

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

# Create uploads directory
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {"message": "AI Video Generator API"}


@app.post("/api/generate-video")
async def generate_video(
    file: Optional[UploadFile] = File(None),
    text_prompt: Optional[str] = Form(None),
    duration: int = Form(5),  # 5 or 10 seconds
    replace_face: bool = Form(False),
    face_image: Optional[UploadFile] = File(None)
):
    """
    Generate video from photo, text, or video input.
    
    Args:
        file: Uploaded photo or video file
        text_prompt: Text prompt for video generation
        duration: Video duration in seconds (5 or 10)
        replace_face: Whether to replace face in generated video
        face_image: User's face image for replacement
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
                # Provide helpful error message for face detection issues
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


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

