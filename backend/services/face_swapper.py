import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess
import os

# Try to import face recognition libraries
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

class FaceSwapper:
    """
    Handles face replacement in videos.
    Uses face recognition (if available) or OpenCV DNN face detection as fallback.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenCV DNN face detector (works without face_recognition)
        self.dnn_net = None
        self._init_opencv_dnn()
        
        # Initialize dlib face detector (optional, if available)
        self.face_detector = None
        self.shape_predictor = None
        
        if DLIB_AVAILABLE:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                predictor_path = "models/shape_predictor_68_face_landmarks.dat"
                if Path(predictor_path).exists():
                    self.shape_predictor = dlib.shape_predictor(predictor_path)
            except:
                pass
    
    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN face detector (works without external dependencies)."""
        try:
            # Try to load OpenCV DNN face detection model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            prototxt_path = model_dir / "deploy.prototxt"
            model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            
            # If models don't exist, we'll use Haar Cascade as fallback
            if prototxt_path.exists() and model_path.exists():
                self.dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
            else:
                self.dnn_net = None
            
            # Always initialize Haar Cascade as fallback (built into OpenCV)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            try:
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    self.face_cascade = None
            except:
                self.face_cascade = None
        except Exception as e:
            print(f"Warning: Could not initialize OpenCV DNN face detector: {e}")
            # Fallback to Haar Cascade
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    self.face_cascade = None
            except:
                self.face_cascade = None
    
    async def swap_face(
        self,
        source_video: Path,
        target_face: Path
    ) -> Path:
        """
        Replace face in source video with target face.
        Works with face_recognition library or OpenCV face detection as fallback.
        
        Args:
            source_video: Path to generated video
            target_face: Path to user's face image
        
        Returns:
            Path to video with replaced face
        """
        # Load target face
        target_img = cv2.imread(str(target_face))
        if target_img is None:
            raise ValueError(f"Could not read face image: {target_face}")
        
        # Extract face from target image
        target_face_region = self._extract_face_region(target_img)
        if target_face_region is None:
            raise ValueError("Could not detect face in target image. Please use a clear front-facing photo.")
        
        # Open source video
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {source_video}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        output_path = self.output_dir / f"face_swapped_{source_video.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in frame
            face_boxes = self._detect_faces(frame)
            
            if face_boxes:
                # Replace the first detected face
                x, y, w, h = face_boxes[0]
                
                # Resize target face to match detected face size
                resized_face = cv2.resize(target_face_region, (w, h))
                
                # Improved face replacement with better blending
                face_region = frame[y:y+h, x:x+w]
                
                # Create a mask for better blending
                mask = np.ones((h, w), dtype=np.float32) * 0.9
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                mask = np.expand_dims(mask, axis=2)
                
                # Blend the faces with mask
                blended = (face_region * (1 - mask) + resized_face * mask).astype(np.uint8)
                frame[y:y+h, x:x+w] = blended
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Re-encode for compatibility
        final_path = self.output_dir / f"final_face_swapped_{source_video.stem}.mp4"
        self._reencode_video(output_path, final_path, fps)
        
        return final_path
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using available methods.
        Returns list of (x, y, width, height) tuples.
        """
        faces = []
        
        # Try face_recognition first (most accurate)
        if FACE_RECOGNITION_AVAILABLE:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    for top, right, bottom, left in face_locations:
                        faces.append((left, top, right - left, bottom - top))
                    return faces
            except:
                pass
        
        # Fallback to OpenCV DNN
        if self.dnn_net is not None:
            try:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123])
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2 - x1, y2 - y1))
                
                if faces:
                    return faces
            except:
                pass
        
        # Fallback to Haar Cascade
        if hasattr(self, 'face_cascade') and self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in detected_faces:
                    faces.append((x, y, w, h))
            except:
                pass
        
        return faces
    
    def _extract_face_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region from image."""
        face_boxes = self._detect_faces(image)
        if face_boxes:
            x, y, w, h = face_boxes[0]
            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return image[y:y+h, x:x+w]
        return None
    
    
    def _reencode_video(self, input_path: Path, output_path: Path, fps: int):
        """Re-encode video using ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-r', str(fps),
                '-pix_fmt', 'yuv420p',
                '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            import shutil
            shutil.copy(input_path, output_path)

