import React, { useState, useRef } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';
import axios from 'axios';

export default function VoiceRecorder({ onVoiceSent }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await sendAudioToBackend(audioBlob);
        
        // Cleanup tracks to turn off the red recording light on the browser tab
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Microphone access denied or failed:", error);
      alert("Please allow microphone permissions to use voice search.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioToBackend = async (blob) => {
    setIsProcessing(true);
    const formData = new FormData();
    // Append the blob as a file named "voice_note.webm"
    formData.append('audio_file', blob, 'voice_note.webm');
    formData.append('user_id', 'guest_user'); // Replace with actual user context

    try {
      // Send to your FastAPI audio stream endpoint
      const response = await axios.post('/api/v2/chat/voice-stream', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // Pass the backend response (transcript + agent reply) up to the Chat UI
      if (onVoiceSent) {
        onVoiceSent(response.data);
      }
    } catch (error) {
      console.error("Failed to send audio:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  if (isProcessing) {
    return (
      <button disabled className="p-3 rounded-full bg-slate-800 border border-slate-700 text-cyan-500 cursor-not-allowed">
        <Loader2 className="w-5 h-5 animate-spin" />
      </button>
    );
  }

  return (
    <button
      onClick={isRecording ? stopRecording : startRecording}
      className={`p-3 rounded-full transition-all duration-300 border flex items-center justify-center
        ${isRecording 
          ? 'bg-red-500/10 border-red-500/50 text-red-500 animate-pulse shadow-[0_0_15px_rgba(239,68,68,0.3)]' 
          : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-cyan-400 hover:text-cyan-300'
        }`}
      title={isRecording ? "Stop Recording" : "Send Voice Note"}
    >
      {isRecording ? (
        <Square className="w-5 h-5 fill-current" />
      ) : (
        <Mic className="w-5 h-5" />
      )}
    </button>
  );
}
