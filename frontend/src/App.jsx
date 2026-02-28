import { useState, useEffect } from 'react';
import './styles/App.css';
import FileUpload from './components/FileUpload';
import AudioRecorder from './components/AudioRecorder';
import EmotionResult from './components/EmotionResult';
import HistoryTable from './components/HistoryTable';
import { checkHealth, getHistory } from './utils/api';

function App() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    // Check API health on mount
    checkAPIHealth();
    loadHistory();
  }, []);

  const checkAPIHealth = async () => {
    try {
      const health = await checkHealth();
      setHealthStatus(health);
    } catch (error) {
      console.error('API health check failed:', error);
      setHealthStatus({ status: 'unhealthy', model_loaded: false });
    }
  };

  const loadHistory = async () => {
    try {
      const data = await getHistory();
      if (data.success) {
        setHistory(data.predictions);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handlePredictionComplete = (prediction) => {
    setResult(prediction);
    loadHistory();
  };

  const handleLoading = (isLoading) => {
    setLoading(isLoading);
  };

  const handleHistoryUpdate = () => {
    loadHistory();
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="container">
          <h1 className="title">
            <span className="emoji">🎤</span> Speech Emotion Analysis Using Voice
          </h1>
          <p className="subtitle">
            Analyze emotions from voice using advanced AI and machine learning
          </p>

          {/* API Status Badge */}
          <div className="status-badge">
            {healthStatus?.model_loaded ? (
              <span className="badge success">✅ Model Ready</span>
            ) : (
              <span className="badge warning">⚠️ Model Not Loaded</span>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">

          {/* Input Section */}
          <section className="input-section">
            <h2>📊 Analyze Your Voice</h2>
            <p className="section-desc">
              Upload an audio file or record your voice to detect emotions
            </p>

            <div className="input-container">
              {/* File Upload */}
              <div className="input-card">
                <h3>📁 Upload Audio File</h3>
                <FileUpload
                  onPredictionComplete={handlePredictionComplete}
                  onLoading={handleLoading}
                />
              </div>

              {/* Audio Recorder */}
              <div className="input-card">
                <h3>🎙️ Record Audio</h3>
                <AudioRecorder
                  onPredictionComplete={handlePredictionComplete}
                  onLoading={handleLoading}
                />
              </div>
            </div>
          </section>

          {/* Result Section */}
          {(result || loading) && (
            <section className="result-section">
              <h2>🎯 Analysis Result</h2>
              <EmotionResult result={result} loading={loading} />
            </section>
          )}

          {/* History Section */}
          <section className="history-section">
            <h2>📜 Prediction History</h2>
            <HistoryTable
              history={history}
              onHistoryUpdate={handleHistoryUpdate}
            />
          </section>

        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="container">
          <p>Speech Emotion Analysis</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
