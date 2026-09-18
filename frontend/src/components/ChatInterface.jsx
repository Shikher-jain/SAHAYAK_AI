import React, { useState } from 'react';
import apiClient from '../api/client';

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [cacheStatus, setCacheStatus] = useState('MISS');

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    
    try {
      const response = await apiClient.post('/chat/orchestrate', {
        query: input,
        session_id: 'session-id',
        mode: 'text'
      });
      
      setCacheStatus(response.headers['x-cache-status'] || 'MISS');
      
      const assistantMsg = { role: 'assistant', content: response.data.reply };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (error) {
      console.error('Chat error', error);
    }
    setInput('');
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 text-white">
      <div className="flex justify-between items-center p-4 border-b border-slate-800">
        <h2 className="text-xl font-bold">Sahayak AI</h2>
        <div className={`px-3 py-1 rounded text-sm font-semibold ${
          cacheStatus === 'HIT' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
        }`}>
          Cache {cacheStatus}
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`p-3 rounded-lg max-w-[80%] ${
            msg.role === 'user' ? 'bg-blue-600 ml-auto' : 'bg-slate-800 mr-auto'
          }`}>
            {msg.content}
          </div>
        ))}
      </div>
      
      <div className="p-4 border-t border-slate-800">
        <div className="flex gap-2">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            className="flex-1 bg-slate-800 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your message..."
          />
          <button onClick={sendMessage} className="bg-blue-600 px-4 py-2 rounded font-semibold hover:bg-blue-700">
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
