import { useCallback, useEffect, useRef, useState } from 'react'
import { Mic, Square, Loader2 } from 'lucide-react'
import { sendVoiceStream } from '../api/v2_client'

/**
 * VoiceRecorder — native-browser voice capture for Sahayak v2.
 *
 * Uses the HTML5 MediaRecorder API (MediaRecorder + navigator.mediaDevices) —
 * NO third-party audio lib (per the zero-heavy-RAM / zero-local-ML spec). The
 * captured webm blob is streamed to `POST /api/v2/chat/voice-stream` where the
 * backend pipes it through Groq Whisper (hosted) and the supervisor graph.
 *
 * Memory hygiene (Rule: kill the red-dot, drop the stream):
 *   - every captured blob chunk is pushed to a ref array (never daisy-chained
 *     into component state → no React re-render per 250 ms audio slice)
 *   - on stop, ALL MediaStream tracks are explicitly stopped so the tab's
 *     recording indicator + mic hardware are released immediately (this is the
 *     #1 leak the reviews flag — browsers keep the mic light on until tracks
 *     die)
 *   - the assembled Blob is sent and then chunk refs are cleared
 */
export default function VoiceRecorder({
  onTranscribed,   // (transcript: string) => void
  disabled = false,
}) {
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])

  const stopTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [])

  useEffect(() => {
    // Defensive teardown if the user closes the tab mid-recording.
    return () => stopTracks()
  }, [stopTracks])

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      })
      streamRef.current = stream

      const mime = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/ogg')
          ? 'audio/ogg'
          : ''
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
      recorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })
        chunksRef.current = []
        stopTracks()
        setIsRecording(false)
        if (blob.size > 0) {
          try {
            setIsProcessing(true)
            const transcript = await sendVoiceStream(blob, 'voice.webm')
            if (onTranscribed) onTranscribed(transcript)
          } finally {
            setIsProcessing(false)
          }
        }
      }

      recorder.start()
      setIsRecording(true)
    } catch (e) {
      console.error('Mic access denied or unavailable:', e)
      alert('Microphone access is required for voice input.')
    }
  }, [onTranscribed, stopTracks])

  const stop = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()
    }
  }, [])

  return (
    <button
      type="button"
      onClick={isRecording ? stop : start}
      disabled={disabled || isProcessing}
      title={isRecording ? 'Stop and transcribe' : 'Record a voice note'}
      className={`shrink-0 w-11 h-11 rounded-full flex items-center justify-center border transition-colors ${
        isRecording
          ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse'
          : 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20 hover:border-cyan-400/60'
      } disabled:opacity-40`}
    >
      {isProcessing ? (
        <Loader2 className="w-5 h-5 animate-spin" />
      ) : isRecording ? (
        <Square className="w-4 h-4 fill-current" />
      ) : (
        <Mic className="w-5 h-5" />
      )}
    </button>
  )
}
