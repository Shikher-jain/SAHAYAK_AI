import React, { useState, useRef } from 'react';
import { 
  UploadCloud, FileText, Globe, AlignLeft, CheckCircle2, 
  AlertCircle, X, ArrowRight, Layers, FileCode, Mic, MicOff, Square
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';

export const Upload = () => {
  const { showSuccess, showError, setCurrentPage } = useAppContext();
  const [activeTab, setActiveTab]       = useState('files');
  const [files, setFiles]               = useState([]);
  const [urlInput, setUrlInput]         = useState('');
  const [textTitle, setTextTitle]       = useState('');
  const [textContent, setTextContent]   = useState('');
  const [loading, setLoading]           = useState(false);
  const [result, setResult]             = useState(null);
  const [isDragging, setIsDragging]     = useState(false);
  const fileInputRef = useRef(null);

  // ─── Audio recording state ─────────────────────────────────────────────────
  const [recording, setRecording]       = useState(false);
  const [audioBlob, setAudioBlob]       = useState(null);
  const [audioUrl, setAudioUrl]         = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef   = useRef([]);

  const tabs = [
    { id: 'files', label: 'Document Files',    icon: FileText,   desc: 'PDF, TXT, MD, CSV, code files' },
    { id: 'audio', label: 'Record Audio',       icon: Mic,        desc: 'Record voice notes or lectures' },
    { id: 'url',   label: 'Web Article / URL',  icon: Globe,      desc: 'Public website or documentation link' },
    { id: 'text',  label: 'Raw Notes / Text',   icon: AlignLeft,  desc: 'Paste syllabus chapters or notes' },
  ];

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      setFiles((prev) => [...prev, ...droppedFiles]);
    }
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  // ─── Audio recording helpers ────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mr.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch {
      showError('Microphone access denied. Please allow microphone in browser settings.');
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const clearRecording = () => {
    setAudioBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
  };

  const handleIngest = async () => {
    setLoading(true);
    setResult(null);

    let ok = false;
    let data = null;
    let error = null;

    try {
      if (activeTab === 'files') {
        if (files.length === 0) return;
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        const res = await callBackend('post', '/ingest/batch', formData);
        ok = res.ok; data = res.data; error = res.error;
      } else if (activeTab === 'audio') {
        if (!audioBlob) return;
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');
        const res = await callBackend('post', '/ingest/audio', formData);
        ok = res.ok; data = res.data; error = res.error;
      } else if (activeTab === 'url') {
        if (!urlInput.trim()) return;
        const res = await callBackend('post', '/ingest/url', { url: urlInput.trim() });
        ok = res.ok; data = res.data; error = res.error;
      } else if (activeTab === 'text') {
        if (!textContent.trim()) return;
        const res = await callBackend('post', '/ingest/text', { text: textContent.trim(), title: textTitle.trim() || 'Untitled Note', content: textContent.trim() });
        ok = res.ok; data = res.data; error = res.error;
      }


      if (ok) {
        setResult({
          success: true,
          message: 'Document successfully ingested and embedded into vector memory!',
          details: data,
        });
        showSuccess('Ingestion complete! You can now query these documents in Search & Chat.');
        // Reset inputs on success
        if (activeTab === 'files') setFiles([]);
        if (activeTab === 'url') setUrlInput('');
        if (activeTab === 'text') { setTextTitle(''); setTextContent(''); }
      } else {
        setResult({
          success: false,
          message: error || 'Ingestion failed. Please check the document format or connection.',
          details: data,
        });
        showError(error || 'Failed to ingest document');
      }
    } catch (e) {
      setResult({
        success: false,
        message: e.message || 'An unexpected error occurred during ingestion.',
      });
      showError(e.message || 'Ingestion error');
    } finally {
      setLoading(false);
    }
  };

  const isSubmitDisabled = () => {
    if (loading) return true;
    if (activeTab === 'files') return files.length === 0;
    if (activeTab === 'audio') return !audioBlob;
    if (activeTab === 'url')   return !urlInput.trim();
    if (activeTab === 'text')  return !textContent.trim();
    return true;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Knowledge Ingestion"
        subtitle="Transform any document, lecture note, or web article into vector embeddings for intelligent RAG search and AI mentorship."
        badge={<Badge variant="primary" size="md">Multimodal</Badge>}
      />

      {/* Tabs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setResult(null); }}
              className={`
                flex items-start gap-3.5 p-4 rounded-2xl border text-left transition-all
                ${isActive 
                  ? 'bg-indigo-50/80 dark:bg-indigo-950/50 border-indigo-300 dark:border-indigo-700 shadow-sm' 
                  : 'bg-white dark:bg-slate-900 border-slate-200/80 dark:border-slate-800/80 hover:border-slate-300 dark:hover:border-slate-700'}
              `}
            >
              <div className={`p-2.5 rounded-xl shrink-0 ${isActive ? 'bg-indigo-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-500'}`}>
                <Icon size={20} />
              </div>
              <div className="min-w-0">
                <h4 className={`text-xs font-bold ${isActive ? 'text-indigo-950 dark:text-indigo-200' : 'text-slate-900 dark:text-slate-200'}`}>
                  {tab.label}
                </h4>
                <p className="text-[11px] text-slate-400 mt-0.5 leading-snug">
                  {tab.desc}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Upload Zone Card */}
      <Card className="p-6 sm:p-8">
        {activeTab === 'files' && (
          <div className="space-y-6">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`
                border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-all
                ${isDragging 
                  ? 'border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/40 scale-[1.01]' 
                  : 'border-slate-200 dark:border-slate-800 hover:border-indigo-400 dark:hover:border-indigo-600 bg-slate-50/40 dark:bg-slate-950/30'}
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={handleFileChange}
                accept=".pdf,.docx,.txt,.md,.csv,.json"
              />
              <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center shadow-sm">
                <UploadCloud size={28} />
              </div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-1">
                Choose files or drag and drop here
              </h3>
              <p className="text-xs text-slate-400 max-w-sm mx-auto mb-4">
                Supported formats: PDF, DOCX, TXT, Markdown, CSV. Files are automatically cleaned, chunked, and vectorized.
              </p>
              <Button size="sm" variant="outline" type="button" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                Browse Files
              </Button>
            </div>

            {/* Selected files list */}
            {files.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs font-bold text-slate-700 dark:text-slate-300 px-1">
                  <span>Selected Files ({files.length})</span>
                  <button
                    type="button"
                    onClick={() => setFiles([])}
                    className="text-rose-500 hover:underline text-[11px] font-semibold"
                  >
                    Clear All
                  </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {files.map((file, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 text-xs"
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <FileCode size={16} className="text-indigo-600 shrink-0" />
                        <div className="min-w-0">
                          <p className="font-semibold text-slate-900 dark:text-slate-100 truncate">{file.name}</p>
                          <p className="text-[10px] text-slate-400">{formatFileSize(file.size)}</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeFile(idx)}
                        className="text-slate-400 hover:text-rose-500 p-1 rounded-md"
                        aria-label="Remove file"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'audio' && (
          <div className="space-y-6">
            <div className="text-center space-y-3">
              <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center transition-all shadow-inner ${
                recording
                  ? 'bg-rose-100 dark:bg-rose-950/50 ring-4 ring-rose-400 ring-offset-4 dark:ring-offset-slate-900 animate-pulse'
                  : 'bg-slate-100 dark:bg-slate-800'
              }`}>
                {recording ? <MicOff size={36} className="text-rose-600" /> : <Mic size={36} className="text-slate-500" />}
              </div>
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                {recording ? '🔴 Recording in progress…' : audioBlob ? '✅ Recording captured' : 'Click to start recording'}
              </p>
              <p className="text-xs text-slate-400">
                Lectures, voice notes, podcasts — transcribed via Whisper and indexed for Q&A.
              </p>
            </div>

            <div className="flex justify-center gap-4">
              {!recording && !audioBlob && (
                <button
                  type="button"
                  onClick={startRecording}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-sm font-bold shadow transition-all"
                >
                  <Mic size={18} /> Start Recording
                </button>
              )}
              {recording && (
                <button
                  type="button"
                  onClick={stopRecording}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-700 hover:bg-slate-800 text-white text-sm font-bold shadow animate-pulse"
                >
                  <Square size={18} /> Stop
                </button>
              )}
              {audioBlob && !recording && (
                <button
                  type="button"
                  onClick={clearRecording}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-xs text-slate-500 hover:text-rose-500"
                >
                  <X size={14} /> Discard
                </button>
              )}
            </div>

            {audioUrl && (
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80">
                <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">Preview Recording</p>
                <audio src={audioUrl} controls className="w-full h-10" />
              </div>
            )}
          </div>
        )}

        {activeTab === 'url' && (
          <div className="space-y-4">
            <Input
              label="Article or Documentation URL"
              placeholder="https://en.wikipedia.org/wiki/Quantum_computing"
              icon={Globe}
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              helperText="The web scraper will fetch content, extract readable text, and ingest it into your knowledge base."
            />
          </div>
        )}

        {activeTab === 'text' && (
          <div className="space-y-4">
            <Input
              label="Document / Note Title"
              placeholder="e.g. Chapter 4: Photosynthesis & Cellular Respiration"
              value={textTitle}
              onChange={(e) => setTextTitle(e.target.value)}
            />
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Text Content
              </label>
              <textarea
                rows={8}
                placeholder="Paste full text, lecture notes, textbook excerpts, or code documentation here..."
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                className="w-full p-4 rounded-xl text-xs font-mono bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 text-slate-900 dark:text-slate-100 placeholder:text-slate-400"
              />
              <p className="text-[11px] text-slate-400 text-right">
                {textContent.length.toLocaleString()} characters
              </p>
            </div>
          </div>
        )}

        {/* Action Button */}
        <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-400">
            Ingested data will immediately be available for semantic search and Q&A.
          </p>
          <Button
            onClick={handleIngest}
            disabled={isSubmitDisabled()}
            loading={loading}
            size="md"
            icon={Layers}
            className="w-full sm:w-auto"
          >
            {loading ? 'Processing & Vectorizing...' : 'Start Knowledge Ingestion'}
          </Button>
        </div>

        {/* Ingestion Result Box */}
        {result && (
          <div
            className={`
              mt-6 p-5 rounded-2xl border text-xs leading-relaxed animate-fade-in
              ${result.success 
                ? 'bg-emerald-50/70 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/50 text-emerald-900 dark:text-emerald-200' 
                : 'bg-rose-50/70 dark:bg-rose-950/30 border-rose-200 dark:border-rose-900/50 text-rose-900 dark:text-rose-200'}
            `}
          >
            <div className="flex items-start gap-3">
              {result.success ? (
                <CheckCircle2 size={20} className="text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
              ) : (
                <AlertCircle size={20} className="text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
              )}
              <div className="flex-1 min-w-0">
                <h4 className="font-bold text-sm mb-1">{result.message}</h4>
                {result.details && (
                  <pre className="mt-2 p-3 rounded-xl bg-white/60 dark:bg-slate-950/60 font-mono text-[11px] overflow-x-auto text-slate-800 dark:text-slate-200 border border-slate-200/50 dark:border-slate-800/50">
                    {JSON.stringify(result.details, null, 2)}
                  </pre>
                )}
                {result.success && (
                  <div className="mt-4 flex gap-3">
                    <Button
                      size="sm"
                      onClick={() => setCurrentPage('search')}
                      icon={ArrowRight}
                      iconPosition="right"
                    >
                      Ask Questions on this Ingestion
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};
