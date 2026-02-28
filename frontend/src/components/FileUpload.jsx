import { useState } from 'react';
import { predictEmotion } from '../utils/api';

function FileUpload({ onPredictionComplete, onLoading }) {
    const [dragActive, setDragActive] = useState(false);
    const [fileName, setFileName] = useState('');
    const [error, setError] = useState('');

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = async (file) => {
        setError('');
        setFileName(file.name);

        // Validate file type
        const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/flac'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|flac|m4a)$/i)) {
            setError('Invalid file type. Please upload an audio file (WAV, MP3, OGG, FLAC)');
            return;
        }

        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            setError('File too large. Maximum size is 16MB');
            return;
        }

        // Upload and predict
        try {
            onLoading(true);
            const result = await predictEmotion(file);

            if (result.success) {
                onPredictionComplete(result);
            } else {
                setError(result.error || 'Prediction failed');
            }
        } catch (err) {
            console.error('Upload error:', err);
            setError(err.error || 'Failed to upload file. Please ensure the backend is running.');
        } finally {
            onLoading(false);
        }
    };

    return (
        <div className="file-upload">
            <form
                className={`upload-form ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onSubmit={(e) => e.preventDefault()}
            >
                <input
                    type="file"
                    id="file-input"
                    accept=".wav,.mp3,.ogg,.flac,.m4a,audio/*"
                    onChange={handleChange}
                    style={{ display: 'none' }}
                />

                <label htmlFor="file-input" className="upload-label">
                    <div className="upload-icon">📂</div>
                    <p className="upload-text">
                        {fileName ? (
                            <>Selected: <strong>{fileName}</strong></>
                        ) : (
                            <>Drag & drop an audio file here or <span className="click-text">click to browse</span></>
                        )}
                    </p>
                    <p className="upload-hint">Supported formats: WAV, MP3, OGG, FLAC (Max 16MB)</p>
                </label>
            </form>

            {error && (
                <div className="error-message">
                    <span className="error-icon">❌</span> {error}
                </div>
            )}
        </div>
    );
}

export default FileUpload;
