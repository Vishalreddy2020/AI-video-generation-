import React, { useState, useRef, useCallback } from 'react';
import { FiPlus, FiFile, FiSend, FiX, FiImage, FiVideo, FiFileText, FiLoader, FiSquare } from 'react-icons/fi';
import { generateImageOrVideo } from '../services/imageApi';
import './Dashboard.css';

const Dashboard = () => {
  const [uploads, setUploads] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedResult, setGeneratedResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const textAreaRef = useRef(null);
  const abortControllerRef = useRef(null);

  const handleFileSelect = useCallback((files) => {
    Array.from(files).forEach((file) => {
      const id = Date.now() + Math.random();
      const fileObj = {
        id,
        file,
        name: file.name,
        type: file.type,
        size: file.size,
        preview: null,
        progress: 0,
      };

      // Generate preview for images
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setUploads((prev) =>
            prev.map((item) =>
              item.id === id ? { ...item, preview: e.target.result, progress: 100 } : item
            )
          );
        };
        reader.readAsDataURL(file);
      } else {
        // Simulate progress for non-image files
        let progress = 0;
        const interval = setInterval(() => {
          progress += 10;
          setUploads((prev) =>
            prev.map((item) =>
              item.id === id ? { ...item, progress } : item
            )
          );
          if (progress >= 100) {
            clearInterval(interval);
          }
        }, 100);
      }

      setUploads((prev) => [...prev, fileObj]);
    });
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files);
    }
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    const files = [];
    for (let i = 0; i < items.length; i++) {
      if (items[i].kind === 'file') {
        const file = items[i].getAsFile();
        if (file) files.push(file);
      }
    }

    if (files.length > 0) {
      handleFileSelect(files);
    }
  };

  const removeUpload = (id) => {
    setUploads((prev) => prev.filter((item) => item.id !== id));
  };

  const handleSend = async () => {
    // Validate: require at least a prompt
    if (!inputText.trim()) {
      setError('Please enter a prompt describing what you want to generate or edit.');
      return;
    }
    
    setIsGenerating(true);
    setError(null);
    setGeneratedResult(null);
    
    // Create AbortController for cancellation
    abortControllerRef.current = new AbortController();
    
    try {
      const formData = new FormData();
      formData.append('prompt', inputText.trim());
      
      // Add the first uploaded file if available
      if (uploads.length > 0 && uploads[0].file) {
        formData.append('file', uploads[0].file);
      }
      
      // Optional parameters
      formData.append('size', '512x512');
      formData.append('strength', '0.75');
      
      const blob = await generateImageOrVideo(formData, abortControllerRef.current.signal);
      
      // Check if blob is valid
      if (!blob || blob.size === 0) {
        throw new Error('Received empty response from server');
      }
      
      // Create object URL for the generated image/video
      const resultUrl = URL.createObjectURL(blob);
      const isVideo = blob.type.startsWith('video/');
      
      console.log('Generated result:', { 
        type: blob.type, 
        size: blob.size, 
        isVideo,
        url: resultUrl 
      });
      
      setGeneratedResult({
        url: resultUrl,
        type: isVideo ? 'video' : 'image',
        blob: blob
      });
      
      // Clear input after successful generation
      setInputText('');
      // Keep uploads for now, user might want to generate again
      
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Generation cancelled.');
      } else {
        console.error('Generation error:', err);
        setError(err.message || 'Failed to generate. Please check your prompt and try again.');
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsGenerating(false);
      setError('Generation cancelled.');
    }
  };

  const getFileIcon = (type) => {
    if (type.startsWith('image/')) return <FiImage className="w-5 h-5" />;
    if (type.startsWith('video/')) return <FiVideo className="w-5 h-5" />;
    return <FiFileText className="w-5 h-5" />;
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="dashboard-container">
      {/* Background gradient effects */}
      <div className="absolute inset-0 bg-blue-glow pointer-events-none" />
      
      {/* Header with Logo */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1 className="logo-text">immages</h1>
          <p className="logo-subtitle">This is an video generator</p>
        </div>
      </header>
      
      <div className="dashboard-grid">
        {/* Left Side - Upload Panel */}
        <div
          className="upload-panel animate-slide-in"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-header">
            <h2 className="text-xl font-semibold text-white">Media</h2>
          </div>

          <button
            onClick={() => fileInputRef.current?.click()}
            className={`upload-button ${isDragging ? 'dragging' : ''}`}
          >
            <FiPlus className="w-8 h-8 text-electric-blue" />
            <span className="text-sm text-gray-300 mt-2">Upload or Drop</span>
          </button>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileInput}
            accept="image/*,video/*"
          />

          {/* Upload List */}
          <div className="upload-list">
            {uploads.map((item) => (
              <div
                key={item.id}
                className="upload-item animate-fade-in"
              >
                {item.preview ? (
                  <div className="upload-preview">
                    <img
                      src={item.preview}
                      alt={item.name}
                      className="upload-preview-image"
                    />
                    <div className="upload-overlay">
                      <button
                        onClick={() => removeUpload(item.id)}
                        className="remove-button"
                      >
                        <FiX className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="upload-file-info">
                    {getFileIcon(item.type)}
                    <div className="upload-file-details">
                      <p className="upload-file-name">{item.name}</p>
                      <p className="upload-file-size">{formatFileSize(item.size)}</p>
                    </div>
                    <button
                      onClick={() => removeUpload(item.id)}
                      className="remove-button"
                    >
                      <FiX className="w-4 h-4" />
                    </button>
                  </div>
                )}
                
                {item.progress < 100 && (
                  <div className="upload-progress-bar">
                    <div
                      className="upload-progress-fill"
                      style={{ width: `${item.progress}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Center - Main Workspace */}
        <div className="main-workspace">
          {/* Input Area */}
          <div className="input-container animate-fade-in-delay">
            <textarea
              ref={textAreaRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onPaste={handlePaste}
              placeholder="Type your prompt here or paste an image...&#10;&#10;Examples:&#10;• 'make background beach sunset and add text vacation mode'&#10;• 'generate a cat wearing sunglasses'&#10;• 'change shirt to black'"
              className="input-textarea"
              rows={8}
            />
            
            <div className="input-footer">
              <div className="input-actions">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="action-button"
                  title="Upload file"
                  disabled={isGenerating}
                >
                  <FiFile className="w-5 h-5" />
                </button>
              </div>
              
              <div className="generate-actions">
                {isGenerating && (
                  <button
                    onClick={handleStop}
                    className="stop-button"
                    title="Stop generation"
                  >
                    <FiSquare className="w-5 h-5" />
                    <span>Stop</span>
                  </button>
                )}
                <button
                  onClick={handleSend}
                  disabled={!inputText.trim() || isGenerating}
                  className="send-button"
                >
                  {isGenerating ? (
                    <>
                      <FiLoader className="w-5 h-5 animate-spin" />
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <FiSend className="w-5 h-5" />
                      <span>Generate</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-message animate-fade-in">
              <div className="error-content">
                <p className="error-text"> {error}</p>
                <button
                  onClick={() => setError(null)}
                  className="error-close"
                >
                  <FiX className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Generated Result */}
          {generatedResult && (
            <div className="preview-gallery animate-fade-in">
              <div className="gallery-header">
                <h3 className="gallery-title">Generated Result</h3>
                <button
                  onClick={() => {
                    setGeneratedResult(null);
                    if (generatedResult.url) {
                      URL.revokeObjectURL(generatedResult.url);
                    }
                  }}
                  className="close-result-button"
                >
                  <FiX className="w-4 h-4" />
                </button>
              </div>
              <div className="result-container">
                {generatedResult.type === 'video' ? (
                  <video
                    src={generatedResult.url}
                    controls
                    className="result-media"
                    autoPlay
                    loop
                  />
                ) : (
                  <img
                    src={generatedResult.url}
                    alt="Generated"
                    className="result-media"
                  />
                )}
                <div className="result-actions">
                  <a
                    href={generatedResult.url}
                    download={`generated_${Date.now()}.${generatedResult.type === 'video' ? 'mp4' : 'png'}`}
                    className="download-button"
                  >
                    <FiFile className="w-4 h-4" />
                    <span>Download</span>
                  </a>
                </div>
              </div>
            </div>
          )}

          {/* Preview Gallery */}
          {uploads.length > 0 && (
            <div className="preview-gallery animate-fade-in">
              <h3 className="gallery-title">Upload Preview</h3>
              <div className="gallery-grid">
                {uploads.map((item) => (
                  item.preview && (
                    <div
                      key={item.id}
                      className="gallery-item animate-scale-in"
                    >
                      <img
                        src={item.preview}
                        alt={item.name}
                        className="gallery-image"
                      />
                      <div className="gallery-overlay">
                        <button
                          onClick={() => removeUpload(item.id)}
                          className="gallery-remove-button"
                        >
                          <FiX className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
