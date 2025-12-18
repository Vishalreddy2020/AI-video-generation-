# Setup Local AI Video Generation (No API Keys!)

## Overview

This guide will help you set up **local AI video generation** using open-source models. No API keys needed - everything runs on your computer!

## Step-by-Step Setup

### Step 1: Install Dependencies

**Windows:**
```bash
backend\install_local_ai.bat
```

**Manual Installation:**
```bash
cd backend
venv\Scripts\activate

# Install PyTorch (with CUDA if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version (slower):
pip install torch torchvision torchaudio

# Install AI model libraries
pip install diffusers transformers accelerate

# Additional utilities
pip install safetensors imageio imageio-ffmpeg
```

### Step 2: System Requirements

**Minimum:**
- 16GB RAM
- 20GB free disk space (for models)
- CPU (will be slow but works)

**Recommended:**
- NVIDIA GPU with 8GB+ VRAM
- 32GB+ RAM
- 50GB+ free disk space

### Step 3: First Run

1. **Start the backend:**
   ```bash
   cd backend
   venv\Scripts\activate
   python main.py
   ```

2. **Generate your first video:**
   - Upload a photo
   - Enter your prompt
   - Click "Generate Video"

3. **First-time model download:**
   - The model will download automatically (~5GB)
   - This happens only once
   - May take 10-20 minutes depending on internet speed

### Step 4: How It Works

The system uses **Stable Video Diffusion** - an open-source model by Stability AI:
- Converts your image into a video
- Understands motion and animation
- Creates realistic video frames
- No API calls - everything local!

## Performance Expectations

**With GPU (NVIDIA 8GB+ VRAM):**
- Model loading: 1-2 minutes
- Video generation: 1-3 minutes per 5-second video

**With CPU only:**
- Model loading: 5-10 minutes
- Video generation: 10-30 minutes per 5-second video

## Troubleshooting

### Out of Memory Error

If you get CUDA out of memory:
1. Reduce `num_frames` in the code
2. Use CPU mode (slower but works)
3. Close other applications

### Model Download Fails

If model download fails:
1. Check internet connection
2. Try again - downloads can be interrupted
3. Models are cached in `~/.cache/huggingface/`

### Very Slow Generation

- GPU is highly recommended
- CPU mode is 10-20x slower
- Consider using a cloud GPU service if needed

## Advanced: Using Different Models

You can switch models in `local_ai_model.py`:
- **Stable Video Diffusion** (default) - Best quality
- **AnimateDiff** - Good for text-to-video
- **ModelScope** - Alternative option

## Next Steps

1. Install dependencies (Step 1)
2. Restart backend
3. Try generating a video!
4. The first generation will download the model
5. Subsequent generations will be faster

## Support

If you encounter issues:
1. Check that all dependencies installed correctly
2. Verify you have enough disk space
3. Check GPU drivers (if using GPU)
4. Try CPU mode if GPU doesn't work

Happy video generating! 🎬

