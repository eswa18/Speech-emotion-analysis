import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
});

// Health check
export const checkHealth = async () => {
    try {
        const response = await api.get('/health');
        return response.data;
    } catch (error) {
        return { status: 'unhealthy', error: error.message };
    }
};

// Predict emotion from audio file
export const predictEmotion = async (audioFile) => {
    try {
        const formData = new FormData();
        formData.append('audio', audioFile);

        const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        return response.data;
    } catch (error) {
        console.error('API Error:', error);
        throw error.response?.data || { error: 'Failed to connect to the server' };
    }
};

// Get prediction history
export const getHistory = async () => {
    try {
        const response = await api.get('/history');
        return response.data;
    } catch (error) {
        throw error.response?.data || error;
    }
};

// Delete a prediction from history
export const deleteHistory = async (predictionId) => {
    try {
        const response = await api.delete(`/history/${predictionId}`);
        return response.data;
    } catch (error) {
        throw error.response?.data || error;
    }
};

// Get supported emotions
export const getEmotions = async () => {
    try {
        const response = await api.get('/emotions');
        return response.data;
    } catch (error) {
        throw error.response?.data || error;
    }
};

export default api;
