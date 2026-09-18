import React, { useState } from 'react';
import VoiceRecorder from './VoiceRecorder';
import ChatMessage from './ChatMessage';
import { Send } from 'lucide-react';

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');

  const handleNewMessage = (newMessage) => {
    // If it's a raw text submission from the user
    if (newMessage.type === 'text') {
      setMessages((prev) => [...prev, { text: newMessage.content, sender: 'user' }]);
      // Here you would typically also trigger the backend API to get the AI response
    }
    // If it's a processed response from the backend (voice or text API response)
    else if (newMessage.type === 'voice_processed' || newMessage.type === 'ai_response') {
      setMessages((prev) => [...prev, { ...newMessage.data, sender: 'ai' }]);
    }
  };

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    
    handleNewMessage({ type: 'text', content: text });
    setText('');
  };

  const handleVoiceResponse = (backendResponse) => {
    handleNewMessage({
      type: 'voice_processed',
      data: backendResponse
    });
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950">
      {/* Header */}
      <header className="p-4 border-b border-slate-800 bg-slate-900 flex justify-between items-center">
        <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          <span className="text-cyan-400">Sahayak</span> AI
        </h1>
      </header>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 max-w-5xl mx-auto w-full">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-500">
            Start a conversation or send a voice note...
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] ${msg.sender === 'user' ? 'bg-cyan-900/30 text-cyan-50' : 'bg-transparent text-slate-200'} rounded-2xl p-1`}>
                {msg.sender === 'user' ? (
                  <div className="p-3 px-4 leading-relaxed">{msg.text}</div>
                ) : (
                  <ChatMessage messageData={msg} />
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input Area */}
      <div className="w-full bg-slate-900 border-t border-slate-800 p-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <VoiceRecorder onVoiceSent={handleVoiceResponse} />

          <form onSubmit={handleTextSubmit} className="flex-1 relative flex items-center">
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Ask Sahayak AI or record a voice note..."
              className="w-full bg-slate-950 border border-slate-800 rounded-full py-3 px-5 text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
            />
            <button
              type="submit"
              disabled={!text.trim()}
              className="absolute right-2 p-2 rounded-full bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 disabled:opacity-50 disabled:hover:bg-cyan-500/10 transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
