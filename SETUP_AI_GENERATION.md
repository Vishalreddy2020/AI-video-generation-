# Setup High-Level AI Video Generation

## Overview

For **high-level AI video generation** where photos actually "perform" actions, you need to use real generative AI models. This guide shows you how to set it up.

## Option 1: Replicate API (Recommended - Easiest)

Replicate provides access to state-of-the-art AI models without needing a GPU.

### Setup:

1. **Get API Key:**
   - Sign up at https://replicate.com
   - Get your API token from https://replicate.com/account/api-tokens

2. **Install dependencies:**
   ```bash
   cd backend
   venv\Scripts\activate
   pip install replicate requests
   ```

3. **Set environment variable:**
   ```bash
   # Windows
   set REPLICATE_API_TOKEN=your_token_here
   
   # Or create .env file in backend directory:
   REPLICATE_API_TOKEN=your_token_here
   ```

4. **Restart backend:**
   ```bash
   python main.py
   ```

### Models Available:
- **Stable Video Diffusion** - High-quality image-to-video
- **AnimateDiff** - Text-to-video with animation
- **Zeroscope** - Fast video generation

## Option 2: Stability AI API

### Setup:

1. **Get API Key:**
   - Sign up at https://platform.stability.ai
   - Get your API key

2. **Set environment variable:**
   ```bash
   set STABILITY_API_KEY=your_key_here
   ```

3. **Restart backend**

## Option 3: Local Models (Advanced)

For running models locally (requires powerful GPU):

### Requirements:
- NVIDIA GPU with 10GB+ VRAM
- CUDA installed
- Large model files (~10-20GB)

### Setup:

1. **Install dependencies:**
   ```bash
   pip install diffusers transformers accelerate torch torchvision
   ```

2. **Download models:**
   - Stable Video Diffusion from Hugging Face
   - Or AnimateDiff models

3. **Configure model paths in code**

## Current Implementation

The system will automatically:
1. Try Replicate API first (if configured)
2. Try Stability AI API (if configured)
3. Fall back to enhanced local generation

## Testing

Once configured, test with:
- Upload a photo
- Enter prompt: "She needs to throw the flower into the air and a camera needs to appear"
- Generate video

The AI will create actual motion and effects based on your prompt!

## Cost Considerations

- **Replicate**: Pay per generation (~$0.01-0.05 per video)
- **Stability AI**: Credits-based pricing
- **Local**: Free but requires powerful hardware

## Recommended: Start with Replicate

It's the easiest way to get high-quality AI video generation working immediately.

