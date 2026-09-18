import React from 'react'
import { createRoot } from 'react-dom/client'
import 'react-force-graph-2d/dist/react-force-graph-2d.css'
import ChatInterface from './components/ChatInterface.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <ChatInterface />
    </div>
  </React.StrictMode>
)
