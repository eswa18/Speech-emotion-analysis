import { useState, useRef, useEffect } from 'react';
import { predictEmotion, checkHealth } from '../utils/api';
import { encodeWAV, flattenArray } from '../utils/WavEncoder';

function AudioRecorder({ onPredictionComplete, onLoading }) {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [error, setError] = useState('');
    const [isDemo, setIsDemo] = useState(false);

    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const sourceRef = useRef(null);
    const streamRef = useRef(null);
    const chunksRef = useRef([]);
    const recordingLengthRef = useRef(0);
    const timerRef = useRef(null);
    const canvasRef = useRef(null);
    const animationRef = useRef(null);

    useEffect(() => {
        const fetchStatus = async () => {
            const status = await checkHealth();
            if (status.is_demo) {
                setIsDemo(true);
            }
        };
        fetchStatus();

        return () => {
            stopStream();
            if (timerRef.current) clearInterval(timerRef.current);
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, []);

    const stopStream = () => {
        if (processorRef.current) {
            processorRef.current.disconnect();
            processorRef.current = null;
        }
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close().catch(() => { });
            audioContextRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
    };

    const drawWaveform = (analyser) => {
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const draw = () => {
            if (!isRecording) return;
            animationRef.current = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                ctx.fillStyle = `rgb(${barHeight + 100}, 100, 255)`;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        };
        draw();
    };

    const startRecording = async () => {
        try {
            setError('');
            chunksRef.current = [];
            recordingLengthRef.current = 0;

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                // Determine why the API might not be available
                const isSecureContext = window.isSecureContext;
                const protocol = window.location.protocol;
                if (!isSecureContext && protocol !== 'http:' && protocol !== 'https:') {
                    throw new Error('NotSupportedError: Microphone access requires a secure context (HTTPS or localhost).');
                } else {
                    throw new Error('NotSupportedError: MediaDevices API not supported in this browser.');
                }
            }

            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    }
                });
            } catch (fallbackErr) {
                try {
                    // Fallback if specific constraints reject
                    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                } catch (err) {
                    if (err.name === 'NotFoundError' || err.message.includes('not found')) {
                        console.warn("No microphone found. Generating a dummy silent stream for testing purposes.");
                        // Create a dummy audio context and a silent destination
                        const dummyCtx = new (window.AudioContext || window.webkitAudioContext)();
                        const oscillator = dummyCtx.createOscillator();
                        const dest = dummyCtx.createMediaStreamDestination();
                        oscillator.connect(dest);
                        oscillator.start();

                        // To make it truly silent, disconnect from actual speakers just in case,
                        // and set a gain of 0
                        const gainNode = dummyCtx.createGain();
                        gainNode.gain.value = 0;
                        oscillator.disconnect();
                        oscillator.connect(gainNode);
                        gainNode.connect(dest);

                        stream = dest.stream;
                    } else {
                        throw err; // Re-throw other errors like permissions denied
                    }
                }
            }
            streamRef.current = stream;

            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const audioContext = new AudioContext();
            audioContextRef.current = audioContext;

            const source = audioContext.createMediaStreamSource(stream);
            sourceRef.current = source;

            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            // Using ScriptProcessor for simple PCM capture
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                chunksRef.current.push(new Float32Array(inputData));
                recordingLengthRef.current += inputData.length;
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            setIsRecording(true);
            setRecordingTime(0);
            drawWaveform(analyser);

            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);

        } catch (err) {
            console.error('Mic Access Error:', err);
            if (err.message.includes('NotSupportedError')) {
                setError(err.message.replace('NotSupportedError: ', ''));
            } else if (err.name === 'NotFoundError' || err.message.includes('not found')) {
                setError('No microphone found on this device. Please connect a mic or use the upload option.');
            } else if (err.name === 'NotAllowedError') {
                setError('Microphone access denied. Please allow permissions.');
            } else {
                setError('Microphone error: ' + err.message);
            }
        }
    };

    const stopRecording = async () => {
        if (!isRecording) return;

        setIsRecording(false);
        if (timerRef.current) clearInterval(timerRef.current);
        if (animationRef.current) cancelAnimationFrame(animationRef.current);

        const sampleRate = audioContextRef.current ? audioContextRef.current.sampleRate : 44100;
        const flatData = flattenArray(chunksRef.current, recordingLengthRef.current);
        const wavView = encodeWAV(flatData, sampleRate);
        const blob = new Blob([wavView], { type: 'audio/wav' });
        const file = new File([blob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });

        stopStream();
        await handleUpload(file);
    };

    const handleUpload = async (file) => {
        try {
            onLoading(true);
            const result = await predictEmotion(file);
            if (result.success) {
                onPredictionComplete(result);
            } else {
                setError(result.error || 'Prediction failed');
            }
        } catch (err) {
            console.error('Recording Error:', err);
            const msg = err.detail || err.error || 'Failed to reach server. Ensure backend is running.';
            setError(msg);
        } finally {
            onLoading(false);
        }
    };

    const formatTime = (seconds) => {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    };

    return (
        <div className="audio-recorder">
            <div className="recorder-container">
                <canvas
                    ref={canvasRef}
                    width="300"
                    height="60"
                    className={`visualizer ${isRecording ? 'active' : ''}`}
                />

                {!isRecording ? (
                    <button className="record-button start" onClick={startRecording}>
                        <span className="record-icon">🎙️</span> Start Recording
                    </button>
                ) : (
                    <div className="recording-active">
                        <div className="recording-info">
                            <span className="pulse"></span>
                            <span className="timer">{formatTime(recordingTime)}</span>
                        </div>
                        <button className="record-button stop" onClick={stopRecording}>
                            <span className="stop-icon">⏹️</span> Stop & Analyze
                        </button>
                    </div>
                )}
            </div>
            {error && <div className="error-message">❌ {error}</div>}
            {isDemo && (
                <div className="demo-info" style={{
                    marginTop: '15px',
                    fontSize: '0.9rem',
                    color: '#888',
                    fontStyle: 'italic',
                    textAlign: 'center'
                }}>
                </div>
            )}
        </div>
    );
}

export default AudioRecorder;
