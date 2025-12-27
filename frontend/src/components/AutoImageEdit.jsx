import React, { useState } from 'react';
import { editImageAuto } from '../services/imageApi';
import './AutoImageEdit.css';

const AutoImageEdit = () => {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [editPrompt, setEditPrompt] = useState('');
  const [strength, setStrength] = useState(0.75);
  const [showMaskPreview, setShowMaskPreview] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [editedImageUrl, setEditedImageUrl] = useState(null);
  const [maskImageUrl, setMaskImageUrl] = useState(null);
  const [progressText, setProgressText] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const url = URL.createObjectURL(file);
      setImagePreviewUrl(url);
      // Clear previous results
      setEditedImageUrl(null);
      setMaskImageUrl(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!imageFile || !editPrompt.trim()) {
      setError('Please upload an image and enter an edit prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setProgressText('Preparing request...');
    setEditedImageUrl(null);
    setMaskImageUrl(null);

    try {
      const formData = new FormData();
      formData.append('image_file', imageFile);
      formData.append('edit_prompt', editPrompt);
      formData.append('strength', strength.toString());
      formData.append('return_mask', showMaskPreview.toString());

      setProgressText('Sending request to AI model...');
      
      const response = await editImageAuto(formData);

      setProgressText('Processing response...');

      // Handle response - could be URLs or base64
      if (response.edited_image_url) {
        setEditedImageUrl(response.edited_image_url);
      } else if (response.edited_image_base64) {
        setEditedImageUrl(`data:image/png;base64,${response.edited_image_base64}`);
      } else {
        throw new Error('No edited image in response');
      }

      if (showMaskPreview && response.mask_image_url) {
        setMaskImageUrl(response.mask_image_url);
      } else if (showMaskPreview && response.mask_image_base64) {
        setMaskImageUrl(`data:image/png;base64,${response.mask_image_base64}`);
      }

      setProgressText('');
    } catch (err) {
      setError(err.message || 'Failed to edit image. Please try again.');
      console.error('Error editing image:', err);
      setProgressText('');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (imageUrl, filename) => {
    if (imageUrl) {
      const a = document.createElement('a');
      a.href = imageUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleUseForVideo = () => {
    if (editedImageUrl) {
      window.dispatchEvent(new CustomEvent('useImageForVideo', {
        detail: { 
          imageUrl: editedImageUrl, 
          imagePath: 'auto_edited_image.png' 
        }
      }));
    }
  };

  return (
    <div className="auto-image-edit">
      <form onSubmit={handleSubmit} className="auto-edit-form">
        <div className="form-section">
          <h2>Automatic Image Edit</h2>
          <p className="section-description">
            Upload an image and describe how you want to edit it. The AI will automatically 
            determine which parts of the image to modify based on your prompt - no manual 
            masking required!
          </p>

          <div className="input-group">
            <label htmlFor="image-upload">Image Upload *</label>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              disabled={loading}
              required
            />
            {imageFile && (
              <p className="file-info">Selected: {imageFile.name}</p>
            )}
          </div>

          {imagePreviewUrl && (
            <div className="image-preview-section">
              <h3>Original Image</h3>
              <div className="image-preview-container">
                <img 
                  src={imagePreviewUrl} 
                  alt="Original" 
                  className="preview-image-original" 
                />
              </div>
            </div>
          )}

          <div className="input-group">
            <label htmlFor="edit-prompt">Edit Prompt *</label>
            <textarea
              id="edit-prompt"
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              placeholder="Describe how you want to edit the image...&#10;&#10;Examples:&#10;- 'change the shirt to black'&#10;- 'make the sky more dramatic with clouds'&#10;- 'add sunglasses to the person'&#10;- 'change the background to a beach'&#10;- 'make it look like a watercolor painting'"
              rows={6}
              required
              disabled={loading}
            />
            <p className="help-text">
              💡 <strong>Tip:</strong> Be specific about what you want to change. The AI will 
              automatically identify and edit the relevant parts.
            </p>
          </div>

          <div className="input-group">
            <label htmlFor="strength">
              Edit Strength: {strength}
            </label>
            <input
              id="strength"
              type="range"
              min="0.6"
              max="0.9"
              step="0.05"
              value={strength}
              onChange={(e) => setStrength(parseFloat(e.target.value))}
              disabled={loading}
            />
            <div className="range-labels">
              <span>Subtle (0.6)</span>
              <span>Strong (0.9)</span>
            </div>
            <p className="help-text">
              Controls how much the image changes. Lower values preserve more of the original, 
              higher values allow more dramatic edits.
            </p>
          </div>

          <div className="input-group checkbox-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showMaskPreview}
                onChange={(e) => setShowMaskPreview(e.target.checked)}
                disabled={loading}
              />
              <span>Show mask preview</span>
            </label>
            <p className="help-text">
              When enabled, the AI will return a preview of the region it selected for editing.
            </p>
          </div>
        </div>

        <button
          type="submit"
          className="edit-btn"
          disabled={loading || !imageFile || !editPrompt.trim()}
        >
          {loading ? 'AI is Editing...' : 'Edit'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>{progressText || 'AI is analyzing your image and prompt...'}</p>
          <p className="loading-subtext">
            Determining what to edit based on your description...
          </p>
          <p className="loading-subtext">This may take 1-3 minutes.</p>
        </div>
      )}

      {(editedImageUrl || maskImageUrl) && (
        <div className="results-section">
          {editedImageUrl && (
            <div className="result-panel">
              <h2>Edited Image</h2>
              <div className="image-preview-container">
                <img 
                  src={editedImageUrl} 
                  alt="Edited" 
                  className="preview-image-result" 
                />
              </div>
              <div className="image-actions">
                <button 
                  onClick={() => handleDownload(editedImageUrl, 'edited_image.png')} 
                  className="download-btn"
                >
                  Download Image
                </button>
                <button 
                  onClick={handleUseForVideo} 
                  className="use-for-video-btn"
                >
                  Use for Video Generation
                </button>
              </div>
            </div>
          )}

          {maskImageUrl && (
            <div className="result-panel mask-panel">
              <h2>Model-Selected Region</h2>
              <p className="mask-description">
                This shows the area the AI identified for editing (white = edit, black = keep)
              </p>
              <div className="image-preview-container mask-preview-container">
                <img 
                  src={maskImageUrl} 
                  alt="Mask" 
                  className="preview-image-mask" 
                />
              </div>
              <button 
                onClick={() => handleDownload(maskImageUrl, 'mask.png')} 
                className="download-btn"
              >
                Download Mask
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AutoImageEdit;

