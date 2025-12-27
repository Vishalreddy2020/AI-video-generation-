/**
 * Image API service for making requests to the backend
 */

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Edit image automatically (AI determines what to edit)
 * @param {FormData} formData - FormData containing:
 *   - image_file: File object
 *   - edit_prompt: string
 *   - strength: float (0.6-0.9)
 *   - return_mask: boolean
 * @returns {Promise<{edited_image_url: string, mask_image_url?: string}>}
 */
export const editImageAuto = async (formData) => {
  try {
    const response = await fetch(`${API_URL}/image/edit/auto`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error in editImageAuto:', error);
    throw error;
  }
};

