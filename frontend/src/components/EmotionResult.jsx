import React, { useEffect, useState } from 'react';

function EmotionResult({ result, loading }) {
    const [animate, setAnimate] = useState(false);

    useEffect(() => {
        if (!loading && result) {
            setAnimate(false);
            const timer = setTimeout(() => setAnimate(true), 100);
            return () => clearTimeout(timer);
        }
    }, [result, loading]);

    if (loading) {
        return (
            <div className="emotion-result loading">
                <div className="loader"></div>
                <p>Analyzing emotion...</p>
            </div>
        );
    }

    if (!result) return null;

    const { emotion, confidence, all_probabilities } = result;

    const emotionEmojis = {
        'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
        'Fear': '😨', 'Surprise': '😲', 'Disgust': '🤢', 'Neutral': '😐'
    };

    const emotionColors = {
        'Happy': '#FFD700', 'Sad': '#4A90E2', 'Angry': '#E74C3C',
        'Fear': '#9B59B6', 'Surprise': '#F39C12', 'Disgust': '#16A085', 'Neutral': '#95A5A6'
    };

    const emoji = emotionEmojis[emotion] || '🎭';
    const color = emotionColors[emotion] || '#3498DB';

    return (
        <div className={`emotion-result-container ${animate ? 'animate' : ''}`}>
            <div className="main-prediction" style={{ borderColor: color }}>
                <div className="prediction-header">
                    <span className="main-emoji" style={{ color }}>{emoji}</span>
                    <div className="prediction-meta">
                        <h2>{emotion} </h2>
                        <span className="confidence-text">{confidence.toFixed(1)}% Confidence</span>
                    </div>
                </div>

                <div className="main-progress-bar">
                    <div
                        className="bar-fill"
                        style={{
                            width: animate ? `${confidence}%` : '0%',
                            backgroundColor: color
                        }}
                    ></div>
                </div>
            </div>

            {all_probabilities && (
                <div className="probability-breakdown">
                    <h3>Detailed Probabilities</h3>
                    <div className="breakdown-list">
                        {Object.entries(all_probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([em, prob]) => (
                                <div key={em} className="breakdown-item">
                                    <div className="item-label">
                                        <span>{emotionEmojis[em]} {em} </span>
                                        <span> {prob.toFixed(1)}%</span>
                                    </div>
                                    <div className="item-bar-bg">
                                        <div
                                            className="item-bar-fill"
                                            style={{
                                                width: animate ? `${prob}%` : '0%',
                                                backgroundColor: emotionColors[em]
                                            }}
                                        ></div>
                                    </div>
                                </div>
                            ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default EmotionResult;
