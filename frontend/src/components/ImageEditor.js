import React, { useState } from 'react';
import axios from 'axios';
import './ImageEditor.css';

const ImageEditor = () => {
  const [sourceImage, setSourceImage] = useState(null);
  const [sourceImageUrl, setSourceImageUrl] = useState(null);
  const [editPrompt, setEditPrompt] = useState('');
  const [strength, setStrength] = useState(0.75);
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const [error, setError] = useState(null);
  const [editedImagePath, setEditedImagePath] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleSourceImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSourceImage(file);
      const url = URL.createObjectURL(file);
      setSourceImageUrl(url);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setImageUrl(null);

    try {
      const formData = new FormData();
      formData.append('source_image', sourceImage);
      formData.append('edit_prompt', editPrompt);
      formData.append('strength', strength);

      const response = await axios.post(
        `${API_URL}/image/edit`,
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
      const filename = response.headers['content-disposition']?.split('filename=')[1]?.replace(/"/g, '') || 'edited_image.png';
      setEditedImagePath(filename);
      
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to edit image');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (imageUrl) {
      const a = document.createElement('a');
      a.href = imageUrl;
      a.download = 'edited_image.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleUseForVideo = () => {
    if (imageUrl && editedImagePath) {
      window.dispatchEvent(new CustomEvent('useImageForVideo', {
        detail: { imageUrl, imagePath: editedImagePath }
      }));
    }
  };

  return (
    <div className="image-editor">
      <form onSubmit={handleSubmit} className="editor-form">
        <div className="form-section">
          <h2>Edit Image with AI</h2>
          <p className="section-description">
            Upload an image and describe how you want to edit it. The AI will automatically determine what parts to change based on your prompt.
          </p>
          
          <div className="input-group">
            <label htmlFor="source-image">Source Image *</label>
            <input
              id="source-image"
              type="file"
              accept="image/*"
              onChange={handleSourceImageChange}
              disabled={loading}
              required
            />
            {sourceImage && (
              <p className="file-info">Selected: {sourceImage.name}</p>
            )}
          </div>

          {sourceImageUrl && (
            <div className="image-preview-small">
              <img src={sourceImageUrl} alt="Source" className="preview-thumbnail" />
            </div>
          )}

          <div className="input-group">
            <label htmlFor="edit-prompt">Edit Prompt *</label>
            <textarea
              id="edit-prompt"
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              placeholder="Describe how you want to edit the image...&#10;&#10;Examples:&#10;- 'change the sky to a beautiful sunset'&#10;- 'make it look like a watercolor painting'&#10;- 'add snow to the scene'&#10;- 'transform it into a cyberpunk style'&#10;- 'change the background to a beach'"
              rows={6}
              required
              disabled={loading}
            />
            <p className="help-text">
              💡 <strong>Tip:</strong> Be specific! The AI will automatically identify and edit the relevant parts based on your description.
            </p>
          </div>

          <div className="input-group">
            <label htmlFor="strength">
              Edit Strength: {strength}
            </label>
            <input
              id="strength"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={strength}
              onChange={(e) => setStrength(parseFloat(e.target.value))}
              disabled={loading}
            />
            <div className="range-labels">
              <span>Subtle (0.0)</span>
              <span>Strong (1.0)</span>
            </div>
            <p className="help-text">
              Lower values (0.3-0.5) = subtle changes, preserves more of the original image.<br/>
              Higher values (0.7-0.9) = more dramatic changes, transforms the image more.
            </p>
          </div>
        </div>

        <button
          type="submit"
          className="generate-btn"
          disabled={loading || !sourceImage || !editPrompt.trim()}
        >
          {loading ? 'AI is Editing Your Image...' : 'Edit Image'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}

      {imageUrl && (
        <div className="image-preview">
          <h2>Edited Image</h2>
          <img src={imageUrl} alt="Edited" className="preview-image" />
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
          <p>AI is analyzing your image and prompt...</p>
          <p className="loading-subtext">Determining what to edit based on your description...</p>
          <p className="loading-subtext">This may take 1-3 minutes.</p>
        </div>
      )}
    </div>
  );
};

export default ImageEditor;
