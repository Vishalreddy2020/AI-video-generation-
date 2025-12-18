import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './VideoGenerator.css';

const VideoGenerator = () => {
  const [inputFile, setInputFile] = useState(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [duration, setDuration] = useState(5);
  const [replaceFace, setReplaceFace] = useState(false);
  const [faceImage, setFaceImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [error, setError] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Test backend connection on component mount
  useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/health`);
        console.log('Backend connection successful:', response.data);
      } catch (err) {
        console.error('Backend connection failed:', err.message);
        setError(`Cannot connect to backend server at ${API_URL}. Make sure the backend is running on port 8000.`);
      }
    };
    testConnection();
  }, [API_URL]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setInputFile(file);
      // Don't clear text prompt - allow both file and text to be used together
    }
  };

  const handleFaceImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFaceImage(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setVideoUrl(null);

    try {
      const formData = new FormData();
      
      if (inputFile) {
        formData.append('file', inputFile);
      }
      
      if (textPrompt) {
        formData.append('text_prompt', textPrompt);
      }
      
      formData.append('duration', duration);
      formData.append('replace_face', replaceFace);
      
      if (replaceFace && faceImage) {
        formData.append('face_image', faceImage);
      }

      const response = await axios.post(
        `${API_URL}/api/generate-video`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          responseType: 'blob',
        }
      );

      // Create blob URL for video
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setVideoUrl(url);
    } catch (err) {
      let errorMessage = 'Failed to generate video';
      
      if (err.code === 'ECONNREFUSED' || err.message.includes('Network Error')) {
        errorMessage = `Cannot connect to backend server at ${API_URL}. Please make sure the backend is running on port 8000.`;
      } else if (err.response) {
        // Server responded with error
        errorMessage = err.response.data?.detail || err.response.data?.message || `Server error: ${err.response.status}`;
      } else if (err.request) {
        // Request made but no response
        errorMessage = `No response from server. Make sure the backend is running at ${API_URL}`;
      } else {
        errorMessage = err.message || 'An unexpected error occurred';
      }
      
      setError(errorMessage);
      console.error('Error details:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (videoUrl) {
      const a = document.createElement('a');
      a.href = videoUrl;
      a.download = `generated_video_${duration}s.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="video-generator">
      <form onSubmit={handleSubmit} className="generator-form">
        <div className="form-section">
          <h2>Input Options</h2>
          
          <div className="input-group">
            <label htmlFor="file-input">Upload Photo or Video</label>
            <input
              id="file-input"
              type="file"
              accept="image/*,video/*"
              onChange={handleFileChange}
              disabled={loading}
            />
            {inputFile && (
              <p className="file-info">Selected: {inputFile.name}</p>
            )}
          </div>

          <div className="divider">AND/OR</div>

          <div className="input-group">
            <label htmlFor="text-input">Additional Instructions / Text Prompt</label>
            <textarea
              id="text-input"
              value={textPrompt}
              onChange={(e) => {
                setTextPrompt(e.target.value);
                // Allow both file and text to be used together
              }}
              placeholder={inputFile 
                ? "Add instructions for the video (e.g., 'make it animated', 'add zoom effect', 'create a cinematic look')..." 
                : "Describe the video you want to generate or add instructions..."}
              rows={4}
              disabled={loading}
            />
            {inputFile && (
              <p className="file-info" style={{fontSize: '0.85rem', color: '#666', marginTop: '0.5rem'}}>
                💡 Tip: You can add instructions to enhance your uploaded photo/video
              </p>
            )}
          </div>
        </div>

        <div className="form-section">
          <h2>Video Settings</h2>
          
          <div className="input-group">
            <label>Duration</label>
            <div className="duration-options">
              <button
                type="button"
                className={`duration-btn ${duration === 5 ? 'active' : ''}`}
                onClick={() => setDuration(5)}
                disabled={loading}
              >
                5 seconds
              </button>
              <button
                type="button"
                className={`duration-btn ${duration === 10 ? 'active' : ''}`}
                onClick={() => setDuration(10)}
                disabled={loading}
              >
                10 seconds
              </button>
            </div>
          </div>

          <div className="input-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={replaceFace}
                onChange={(e) => setReplaceFace(e.target.checked)}
                disabled={loading}
              />
              Replace face in video
            </label>
          </div>

          {replaceFace && (
            <div className="input-group">
              <label htmlFor="face-input">Upload Your Face Image</label>
              <input
                id="face-input"
                type="file"
                accept="image/*"
                onChange={handleFaceImageChange}
                disabled={loading}
              />
              {faceImage && (
                <p className="file-info">Face image: {faceImage.name}</p>
              )}
            </div>
          )}
        </div>

        <button
          type="submit"
          className="generate-btn"
          disabled={loading || (!inputFile && !textPrompt.trim())}
        >
          {loading ? 'Generating Video...' : 'Generate Video'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}

      {videoUrl && (
        <div className="video-preview">
          <h2>Generated Video</h2>
          <video controls src={videoUrl} className="preview-video">
            Your browser does not support the video tag.
          </video>
          <button onClick={handleDownload} className="download-btn">
            Download Video
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Generating your video... This may take a few moments.</p>
        </div>
      )}
    </div>
  );
};

export default VideoGenerator;

