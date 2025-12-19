import React, { useState } from 'react';
import axios from 'axios';
import './ImageGenerator.css';

const ImageGenerator = () => {
  const [prompt, setPrompt] = useState('');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [seed, setSeed] = useState('');
  const [style, setStyle] = useState('');
  const [overlayText, setOverlayText] = useState('');
  const [position, setPosition] = useState('bottom_center');
  const [fontSize, setFontSize] = useState('');
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const [error, setError] = useState(null);
  const [generatedImagePath, setGeneratedImagePath] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setImageUrl(null);

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('size', `${width}x${height}`);
      
      if (seed) {
        formData.append('seed', parseInt(seed));
      }
      
      if (style) {
        formData.append('style', style);
      }
      
      if (overlayText) {
        formData.append('overlay_text', overlayText);
        formData.append('position', position);
        if (fontSize) {
          formData.append('font_size', parseInt(fontSize));
        }
      }

      const response = await axios.post(
        `${API_URL}/image/generate`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          responseType: 'blob',
        }
      );

      // Create blob URL for image
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setImageUrl(url);
      
      // Store image path for use in video generation
      const filename = response.headers['content-disposition']?.split('filename=')[1]?.replace(/"/g, '') || 'generated_image.png';
      setGeneratedImagePath(filename);
      
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate image');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (imageUrl) {
      const a = document.createElement('a');
      a.href = imageUrl;
      a.download = 'generated_image.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleUseForVideo = () => {
    // This will be handled by parent component
    if (imageUrl && generatedImagePath) {
      // Trigger event or callback to pass image to video generator
      window.dispatchEvent(new CustomEvent('useImageForVideo', {
        detail: { imageUrl, imagePath: generatedImagePath }
      }));
    }
  };

  return (
    <div className="image-generator">
      <form onSubmit={handleSubmit} className="generator-form">
        <div className="form-section">
          <h2>Generate Image</h2>
          
          <div className="input-group">
            <label htmlFor="prompt">Image Prompt *</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the image you want to generate..."
              rows={4}
              required
              disabled={loading}
            />
          </div>

          <div className="size-inputs">
            <div className="input-group">
              <label htmlFor="width">Width</label>
              <input
                id="width"
                type="number"
                value={width}
                onChange={(e) => setWidth(parseInt(e.target.value) || 512)}
                min="256"
                max="1024"
                step="8"
                disabled={loading}
              />
            </div>

            <div className="input-group">
              <label htmlFor="height">Height</label>
              <input
                id="height"
                type="number"
                value={height}
                onChange={(e) => setHeight(parseInt(e.target.value) || 512)}
                min="256"
                max="1024"
                step="8"
                disabled={loading}
              />
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="style">Style (Optional)</label>
            <select
              id="style"
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              disabled={loading}
            >
              <option value="">None</option>
              <option value="photorealistic">Photorealistic</option>
              <option value="anime">Anime</option>
              <option value="oil painting">Oil Painting</option>
              <option value="watercolor">Watercolor</option>
              <option value="sketch">Sketch</option>
              <option value="3d render">3D Render</option>
              <option value="cartoon">Cartoon</option>
              <option value="cyberpunk">Cyberpunk</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="seed">Seed (Optional)</label>
            <input
              id="seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Random seed for reproducibility"
              disabled={loading}
            />
          </div>

          <div className="input-group">
            <label htmlFor="overlay-text">Overlay Text (Optional)</label>
            <input
              id="overlay-text"
              type="text"
              value={overlayText}
              onChange={(e) => setOverlayText(e.target.value)}
              placeholder="Text to overlay on image"
              disabled={loading}
            />
          </div>

          {overlayText && (
            <>
              <div className="input-group">
                <label htmlFor="position">Text Position</label>
                <select
                  id="position"
                  value={position}
                  onChange={(e) => setPosition(e.target.value)}
                  disabled={loading}
                >
                  <option value="top_left">Top Left</option>
                  <option value="top_center">Top Center</option>
                  <option value="top_right">Top Right</option>
                  <option value="center_left">Center Left</option>
                  <option value="center">Center</option>
                  <option value="center_right">Center Right</option>
                  <option value="bottom_left">Bottom Left</option>
                  <option value="bottom_center">Bottom Center</option>
                  <option value="bottom_right">Bottom Right</option>
                </select>
              </div>

              <div className="input-group">
                <label htmlFor="font-size">Font Size (Optional)</label>
                <input
                  id="font-size"
                  type="number"
                  value={fontSize}
                  onChange={(e) => setFontSize(e.target.value)}
                  placeholder="Auto-calculated if not specified"
                  min="10"
                  max="200"
                  disabled={loading}
                />
              </div>
            </>
          )}
        </div>

        <button
          type="submit"
          className="generate-btn"
          disabled={loading || !prompt.trim()}
        >
          {loading ? 'Generating Image...' : 'Generate Image'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}

      {imageUrl && (
        <div className="image-preview">
          <h2>Generated Image</h2>
          <img src={imageUrl} alt="Generated" className="preview-image" />
          <div className="image-actions">
            <button onClick={handleDownload} className="download-btn">
              Download Image
            </button>
            <button onClick={handleUseForVideo} className="use-for-video-btn">
              Use for Video Generation
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Generating your image... This may take 1-3 minutes.</p>
        </div>
      )}
    </div>
  );
};

export default ImageGenerator;

