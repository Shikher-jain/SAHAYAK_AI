/**
 * useVoice.js
 * Custom hook that wraps:
 *   - Web Speech API SpeechRecognition → STT (speech-to-text)
 *   - Web Speech API speechSynthesis   → TTS (text-to-speech)
 *
 * No external dependencies needed — all built into modern browsers.
 */
import { useState, useRef, useCallback, useEffect } from 'react';

const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition || null;

export const useVoice = ({ onTranscript, lang = 'en-US' } = {}) => {
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking]   = useState(false);
  const [supported, setSupported] = useState(!!SpeechRec);
  const recRef = useRef(null);

  // ─── STT: Start listening ──────────────────────────────────────────────────
  const startListening = useCallback(() => {
    if (!SpeechRec) return;
    if (listening) return;

    const rec = new SpeechRec();
    rec.lang = lang;
    rec.continuous = false;
    rec.interimResults = false;
    rec.maxAlternatives = 1;

    rec.onstart  = () => setListening(true);
    rec.onend    = () => setListening(false);
    rec.onerror  = ()  => setListening(false);

    rec.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (onTranscript) onTranscript(transcript);
    };

    recRef.current = rec;
    rec.start();
  }, [listening, lang, onTranscript]);

  const stopListening = useCallback(() => {
    recRef.current?.stop();
    setListening(false);
  }, []);

  // ─── TTS: Speak text ───────────────────────────────────────────────────────
  const speak = useCallback((text, { rate = 1, pitch = 1 } = {}) => {
    if (!window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();          // stop any current speech

    const utter = new SpeechSynthesisUtterance(text);
    utter.lang  = lang;
    utter.rate  = rate;
    utter.pitch = pitch;

    utter.onstart = () => setSpeaking(true);
    utter.onend   = () => setSpeaking(false);
    utter.onerror = () => setSpeaking(false);

    window.speechSynthesis.speak(utter);
  }, [lang]);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      recRef.current?.stop();
      window.speechSynthesis?.cancel();
    };
  }, []);

  return { listening, speaking, supported, startListening, stopListening, speak, stopSpeaking };
};
