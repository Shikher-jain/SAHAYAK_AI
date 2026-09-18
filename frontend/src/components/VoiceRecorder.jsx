import React, { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import apiClient from '../api/client';

export default function VoiceRecorder({ onTranscriptionComplete }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const startRecording = async () => {
    setError(null);
    chunksRef.current = [];
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: 'audio/webm' };
      const mediaRecorder = new MediaRecorder(stream, options);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await handleAudioSubmit(audioBlob);
        
        // Cleanup tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(250); // emit chunks every 250ms
      setIsRecording(true);
      
      setRecordingTime(0);
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error('Microphone access denied or unsupported', err);
      setError('Could not access microphone.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    clearInterval(timerRef.current);
  };

  const handleAudioSubmit = async (blob) => {
    setIsProcessing(true);
    const formData = new FormData();
    // In our backend tests we expect 'file' for voice-stream
    formData.append('file', blob, 'voice_query.webm');

    try {
      const response = await apiClient.post('/chat/voice-stream', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      if (onTranscriptionComplete) {
        onTranscriptionComplete(response.data);
      }
    } catch (err) {
      console.error('Audio upload failed', err);
      setError('Failed to process audio.');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="flex flex-col items-center justify-center p-4 bg-slate-900 border border-slate-800 rounded-lg shadow-md max-w-sm w-full mx-auto">
      {error && <div className="text-red-400 text-sm mb-3 text-center">{error}</div>}
      
      <div className="flex items-center gap-4">
        {!isRecording ? (
          <button 
            onClick={startRecording}
            disabled={isProcessing}
            className="flex items-center justify-center w-14 h-14 rounded-full bg-red-500/20 text-red-500 hover:bg-red-500 hover:text-white transition-colors disabled:opacity-50"
            title="Start Recording"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
            </svg>
          </button>
        ) : (
          <button 
            onClick={stopRecording}
            className="flex items-center justify-center w-14 h-14 rounded-full bg-red-600 text-white animate-pulse shadow-[0_0_15px_rgba(220,38,38,0.5)]"
            title="Stop Recording"
          >
            <div className="w-5 h-5 bg-white rounded-sm"></div>
          </button>
        )}
        
        <div className="flex flex-col">
          <span className="text-slate-100 font-mono text-lg tracking-wider">
            {formatTime(recordingTime)}
          </span>
          {isProcessing && <span className="text-cyan-400 text-xs mt-1 animate-pulse">Processing...</span>}
          {isRecording && <span className="text-red-400 text-xs mt-1">Recording</span>}
        </div>
      </div>
    </div>
  );
}

VoiceRecorder.propTypes = {
  onTranscriptionComplete: PropTypes.func
};
