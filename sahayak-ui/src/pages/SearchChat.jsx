import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Send, Bot, User, Sparkles, Copy, Check, Mic, MicOff, Volume2, VolumeX,
  BookOpen, ChevronDown, ChevronRight, Trash2, Download
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Spinner } from '../components/ui/Spinner';
import { RAGPipeline } from '../components/ui/RAGPipeline';
import { useVoice } from '../hooks/useVoice';


export const SearchChat = ({ 
  endpoint = '/search/rag', 
  title = "Search & Grounded Chat", 
  subtitle = "Query your uploaded documents and get accurate answers with source citations.",
  placeholder = "Ask any question from your ingested books or documents...",
  extraPayload = {} 
}) => {
  const { 
    ragSessionId, 
    newRagSession, 
    learningMode, 
    userMode, 
    showSuccess, 
    showError 
  } = useAppContext();

  const [messages, setMessages]             = useState([]);
  const [input, setInput]                   = useState('');
  const [loading, setLoading]               = useState(false);
  const [ragDone, setRagDone]               = useState(false);
  const [copiedIdx, setCopiedIdx]           = useState(null);
  const [searchMode, setSearchMode]         = useState('rag');
  const [expandedSources, setExpandedSources] = useState({});
  const [ttsEnabled, setTtsEnabled]         = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef        = useRef(null);

  // Voice hook — STT writes directly to input, TTS reads assistant answers
  const { 
    listening, speaking, supported,
    startListening, stopListening,
    speak, stopSpeaking
  } = useVoice({
    onTranscript: (text) => {
      setInput((prev) => (prev ? `${prev} ${text}` : text));
      // Auto-focus input so user sees the transcribed text
      inputRef.current?.focus();
    }
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => { scrollToBottom(); }, [messages, loading]);

  const toggleSourceExpand = (msgIdx) => {
    setExpandedSources((prev) => ({ ...prev, [msgIdx]: !prev[msgIdx] }));
  };

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    showSuccess('Copied to clipboard');
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  const handleClear = () => {
    setMessages([]);
    newRagSession();
    showSuccess('Started a new conversation session');
  };

  const handleExport = () => {
    if (messages.length === 0) return;
    const text = messages.map(m => `[${m.role.toUpperCase()}]:\n${m.content}\n`).join('\n---\n\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `sahayak-chat-${Date.now()}.txt`; a.click();
    URL.revokeObjectURL(url);
    showSuccess('Chat transcript exported');
  };

  // Toggle mic — if already listening, stop; otherwise start
  const handleMicClick = useCallback(() => {
    if (!supported) { showError('Voice input is not supported in this browser.'); return; }
    if (listening) stopListening();
    else startListening();
  }, [listening, supported, startListening, stopListening, showError]);

  const handleSend = async (textToSend) => {
    const queryText = (textToSend || input).trim();
    if (!queryText || loading) return;

    setInput('');
    setRagDone(false);
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    setMessages((prev) => [...prev, {
      id: `u_${Date.now()}`, role: 'user', content: queryText, timestamp: timeStr
    }]);
    setLoading(true);

    const actualEndpoint = searchMode === 'vector' ? '/search/vector' : endpoint;
    const payload = {
      query: queryText, message: queryText,
      top_k: 5, session_id: ragSessionId,
      learning_mode: learningMode, user_mode: userMode,
      ...extraPayload
    };

    const { ok, data, error } = await callBackend('post', actualEndpoint, payload);
    const respTimeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setRagDone(true);

    if (ok && data) {
      const answer = data.answer || data.response 
        || (Array.isArray(data.results) ? data.results.map(r => r.text || r.content).join('\n\n') : 'Here is the relevant information from your library.');
      const assistantMessage = {
        id: `a_${Date.now()}`, role: 'assistant',
        content: answer,
        sources:   data.sources   || data.context || data.results || [],
        followUps: data.follow_ups || data.suggestions || data.recommended_questions || [],
        timestamp: respTimeStr,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Speak the answer if TTS enabled
      if (ttsEnabled) {
        speak(answer);
      }
    } else {
      setMessages((prev) => [...prev, {
        id: `e_${Date.now()}`, role: 'error',
        content: error || 'Failed to retrieve answer. Please make sure documents are indexed and backend is running.',
        timestamp: respTimeStr,
      }]);
      showError(error || 'Search query failed');
    }

    setLoading(false);
    // Keep ragDone visible briefly, then hide pipeline
    setTimeout(() => setRagDone(false), 3000);
  };

  const quickStarters = [
    "Summarize the key takeaways from my uploaded documents",
    "Explain the core concept in simple terms with examples",
    "What are the main formulas or definitions mentioned?",
    "Create 3 practice quiz questions based on the content"
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)] max-w-5xl mx-auto space-y-4 animate-fade-in text-left">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 shrink-0">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            {title}
            <Badge variant="primary" size="sm">Persona: {userMode}</Badge>
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{subtitle}</p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* RAG / Vector mode switcher */}
          <div className="flex p-0.5 bg-slate-200/70 dark:bg-slate-800 rounded-xl text-xs font-semibold">
            <button
              onClick={() => setSearchMode('rag')}
              className={`px-3 py-1.5 rounded-lg transition-all ${searchMode === 'rag' ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-500'}`}
              title="Full RAG Generation with Synthesized Answers"
            >Grounded RAG</button>
            <button
              onClick={() => setSearchMode('vector')}
              className={`px-3 py-1.5 rounded-lg transition-all ${searchMode === 'vector' ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-500'}`}
              title="Direct Vector Semantic Match Chunks"
            >Vector Match</button>
          </div>

          {/* TTS toggle */}
          <button
            onClick={() => { setTtsEnabled((v) => !v); if (speaking) stopSpeaking(); }}
            title={ttsEnabled ? 'Disable voice answer' : 'Enable voice answer (TTS)'}
            className={`p-2 rounded-xl border transition-all ${ttsEnabled ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-700 text-emerald-600 dark:text-emerald-400' : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500'}`}
          >
            {ttsEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>

          {messages.length > 0 && (
            <>
              <Button variant="outline" size="sm" icon={Download} onClick={handleExport} title="Export Transcript" />
              <Button variant="ghost"   size="sm" icon={Trash2}   onClick={handleClear}  title="Clear Chat" />
            </>
          )}
        </div>
      </div>

      {/* RAG Pipeline visualization (shown while loading or just after done) */}
      {(loading || ragDone) && (
        <div className="shrink-0">
          <RAGPipeline active={loading} done={ragDone && !loading} />
        </div>
      )}

      {/* Main Chat Container */}
      <Card className="flex-1 flex flex-col p-0 overflow-hidden border-slate-200/90 dark:border-slate-800/90 shadow-md">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-6">
              <div className="w-16 h-16 rounded-3xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center shadow-inner">
                <Sparkles size={32} />
              </div>
              <div className="max-w-md space-y-2">
                <h3 className="text-base font-bold text-slate-900 dark:text-white">Knowledge-Grounded AI Assistant</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Ask questions in your natural language. Sahayak retrieves the most relevant paragraphs and generates structured explanations with direct references.
                </p>
                {supported && (
                  <p className="text-xs text-indigo-500 dark:text-indigo-400">
                    🎙️ Click the microphone button to speak your question
                  </p>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-xl w-full">
                {quickStarters.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(q)}
                    className="p-3 text-left rounded-xl bg-slate-50 dark:bg-slate-800/60 hover:bg-indigo-50 dark:hover:bg-indigo-950/40 border border-slate-200/80 dark:border-slate-700/60 hover:border-indigo-200 dark:hover:border-indigo-800 text-xs text-slate-700 dark:text-slate-300 transition-all duration-150 leading-snug group"
                  >
                    <span className="font-medium group-hover:text-indigo-600 dark:group-hover:text-indigo-400">"{q}"</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => {
              const isUser  = msg.role === 'user';
              const isError = msg.role === 'error';
              return (
                <div key={msg.id || idx} className={`flex gap-3 sm:gap-4 max-w-4xl ${isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}>
                  {/* Avatar */}
                  <div className={`w-8 h-8 sm:w-9 sm:h-9 rounded-xl flex items-center justify-center shrink-0 font-bold text-xs shadow-sm
                    ${isUser ? 'bg-indigo-600 text-white' : isError ? 'bg-rose-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 border border-slate-200/60 dark:border-slate-700/60'}`}
                  >
                    {isUser ? <User size={16} /> : isError ? '!' : <Bot size={18} />}
                  </div>

                  {/* Bubble */}
                  <div className={`space-y-2 max-w-[85%] ${isUser ? 'text-right' : 'text-left'}`}>
                    <div className={`p-4 rounded-2xl text-xs sm:text-sm leading-relaxed
                      ${isUser
                        ? 'bg-indigo-600 text-white rounded-tr-none shadow-sm'
                        : isError
                          ? 'bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 text-rose-800 dark:text-rose-200 rounded-tl-none'
                          : 'bg-slate-100/90 dark:bg-slate-800/80 text-slate-900 dark:text-slate-100 rounded-tl-none border border-slate-200/60 dark:border-slate-700/60'}`}
                    >
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>

                    {/* Actions for assistant */}
                    {!isUser && !isError && (
                      <div className="flex items-center gap-3 pt-1 text-[11px] text-slate-400 px-1">
                        <span>{msg.timestamp}</span>
                        <button onClick={() => handleCopy(msg.content, idx)} className="hover:text-slate-700 dark:hover:text-slate-200 flex items-center gap-1 transition-colors">
                          {copiedIdx === idx ? <Check size={12} className="text-emerald-500" /> : <Copy size={12} />}
                          <span>{copiedIdx === idx ? 'Copied' : 'Copy'}</span>
                        </button>
                        {/* Speak this message */}
                        <button
                          onClick={() => speaking ? stopSpeaking() : speak(msg.content)}
                          title={speaking ? 'Stop speaking' : 'Read aloud'}
                          className="hover:text-emerald-600 dark:hover:text-emerald-400 flex items-center gap-1 transition-colors"
                        >
                          {speaking ? <VolumeX size={12} /> : <Volume2 size={12} />}
                          <span>{speaking ? 'Stop' : 'Read'}</span>
                        </button>
                      </div>
                    )}

                    {/* Sources */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800/80 text-left">
                        <button
                          onClick={() => toggleSourceExpand(idx)}
                          className="flex items-center justify-between w-full text-xs font-bold text-slate-700 dark:text-slate-300"
                        >
                          <span className="flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400">
                            <BookOpen size={14} />
                            Sources Cited ({msg.sources.length})
                          </span>
                          {expandedSources[idx] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        </button>
                        {expandedSources[idx] && (
                          <div className="mt-2.5 space-y-2 text-[11px] text-slate-600 dark:text-slate-400">
                            {msg.sources.map((src, sIdx) => (
                              <div key={sIdx} className="p-2.5 rounded-lg bg-white dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800/60">
                                <div className="flex items-center justify-between font-semibold text-slate-900 dark:text-slate-200 mb-1">
                                  <span>{src.source || src.title || src.filename || `Reference #${sIdx + 1}`}</span>
                                  {src.score && (
                                    <span className="text-[10px] px-1.5 rounded bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400">
                                      Match: {(src.score * 100).toFixed(0)}%
                                    </span>
                                  )}
                                </div>
                                <p className="font-mono text-[10px] text-slate-500 line-clamp-3 leading-tight">
                                  {src.text || src.content || src.snippet || JSON.stringify(src)}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Follow-up prompts */}
                    {msg.followUps && msg.followUps.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1 text-left">
                        {msg.followUps.map((q, fIdx) => (
                          <button
                            key={fIdx}
                            onClick={() => handleSend(q)}
                            className="text-xs text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/50 hover:bg-indigo-100 dark:hover:bg-indigo-900/50 px-3 py-1.5 rounded-full border border-indigo-200/50 dark:border-indigo-800/50 transition-colors"
                          >
                            + {q}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Typing indicator */}
          {loading && (
            <div className="flex gap-3 max-w-4xl mr-auto animate-fade-in">
              <div className="w-8 h-8 rounded-xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-indigo-600 shrink-0">
                <Spinner size="sm" />
              </div>
              <div className="p-3.5 bg-slate-100 dark:bg-slate-800/60 rounded-2xl rounded-tl-none border border-slate-200/60 dark:border-slate-700/60 flex items-center gap-2 text-xs text-slate-500">
                <span>Retrieving knowledge and formulating response…</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input form */}
        <form
          onSubmit={(e) => { e.preventDefault(); handleSend(); }}
          className="p-3 sm:p-4 border-t border-slate-200/80 dark:border-slate-800/80 bg-slate-50/70 dark:bg-slate-950/60 flex items-center gap-2"
        >
          {/* Mic button */}
          {supported && (
            <button
              type="button"
              onClick={handleMicClick}
              title={listening ? 'Stop recording' : 'Speak your question (STT)'}
              className={`p-2.5 rounded-xl border transition-all shrink-0
                ${listening
                  ? 'bg-rose-500 border-rose-500 text-white shadow-lg shadow-rose-500/30 animate-pulse'
                  : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 hover:text-indigo-600 hover:border-indigo-300'
                }`}
            >
              {listening ? <MicOff size={18} /> : <Mic size={18} />}
            </button>
          )}

          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={listening ? '🎙️ Listening… speak now' : placeholder}
            disabled={loading}
            className="flex-1 px-4 py-2.5 rounded-xl text-xs sm:text-sm bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 text-slate-900 dark:text-slate-100 placeholder:text-slate-400"
          />

          <Button
            type="submit"
            disabled={!input.trim() || loading}
            size="md"
            icon={Send}
            className="px-4"
          >
            <span className="hidden sm:inline">Ask</span>
          </Button>
        </form>
      </Card>
    </div>
  );
};
