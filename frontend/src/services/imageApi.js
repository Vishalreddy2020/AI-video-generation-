/**
 * Image API service for making requests to the backend
 */

/**
 * API_URL is the base URL for the backend API.
 * If the REACT_APP_API_URL environment variable is set, it will use that.
 * Otherwise, it defaults to 'http://localhost:8000'.
 */
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * generateImageOrVideo
 * 
 * Unified endpoint that plans and executes AI generation tasks.
 * Can generate images from text, edit images, or generate videos.
 * 
 * @param {FormData} formData - Contains:
 *   - prompt: Text description of what to generate/edit
 *   - file: (optional) Image/video file to edit
 *   - size: (optional) Image size like "512x512"
 *   - duration: (optional) Video duration in seconds
 *   - strength: (optional) Edit strength (0.0-1.0)
 *   - style: (optional) Style hint
 *   - seed: (optional) Random seed
 * @param {AbortSignal} signal - Optional AbortSignal to cancel the request
 * 
 * @returns {Promise<Blob>} - The generated image or video file
 */
export const generateImageOrVideo = async (formData, signal = null) => {
  try {
    console.log('Sending request to:', `${API_URL}/api/generate`);
    
    const response = await fetch(`${API_URL}/api/generate`, {
      method: 'POST',
      body: formData,
      signal: signal, // Add abort signal support
    });

    console.log('Response status:', response.status, response.statusText);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      // Try to get error message from response
      let errorMessage = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorMessage;
      } catch (e) {
        // If response is not JSON, try to get text
        try {
          const errorText = await response.text();
          if (errorText) errorMessage = errorText;
        } catch (e2) {
          // Ignore
        }
      }
      throw new Error(errorMessage);
    }

    // Get the blob (image or video file)
    const blob = await response.blob();
    console.log('Received blob:', { 
      type: blob.type, 
      size: blob.size,
      sizeKB: (blob.size / 1024).toFixed(2) 
    });
    
    if (blob.size === 0) {
      throw new Error('Received empty file from server');
    }
    
    return blob;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw error; // Re-throw abort errors
    }
    console.error('Error in generateImageOrVideo:', error);
    throw error;
  }
};

/**
 * editImageAuto
 *
 * This function sends a POST request to the backend to perform an automatic image edit.
 * The AI on the backend decides how to modify the image, based on the given edit prompt.
 * 
 * @param {FormData} formData - Contains these fields:
 *   - image_file: The image file to be edited.
 *   - edit_prompt: A text description (prompt) for what edit to make.
 *   - strength: A float (between 0.6 and 0.9) that controls how strongly the edit is applied.
 *   - return_mask: (boolean) Whether to return a mask image showing which parts were edited.
 * 
 * @returns {Promise<{edited_image_url: string, mask_image_url?: string}>}
 *   - Returns a Promise that resolves to an object containing the URL of the edited image.
 *   - Optionally includes a mask_image_url if 'return_mask' was true.
 * 
 * How it works:
 * 1. Makes a POST request to `${API_URL}/image/edit/auto` with the given FormData.
 *    The browser sets the correct Content-Type for FormData automatically.
 * 2. Checks the response; if it is not OK, attempts to extract an error message from the response.
 *    If no message is found, throws a generic HTTP error with the status code.
 * 3. If successful, parses the JSON response and returns it.
 * 4. Catches and logs any error that occurs during the request or response handling,
 *    then rethrows the error for the caller to handle.
 */
export const editImageAuto = async (formData) => {
  try {
    const response = await fetch(`${API_URL}/image/edit/auto`, {
      method: 'POST',
      body: formData,
      // Content-Type header is NOT set manually so the browser can include boundary info with FormData
    });

    if (!response.ok) {
      // Try to read error details from the response JSON, fallback to empty object if not readable
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    // On success, parse and return the JSON response from the backend
    const data = await response.json();
    return data;
  } catch (error) {
    // Log the error for debugging and rethrow it for further handling
    console.error('Error in editImageAuto:', error);
    throw error;
  }
};

