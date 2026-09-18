import React, { useState } from 'react';
import CitationViewer from './CitationViewer';
import { Target } from 'lucide-react';

export default function ChatMessage({ messageData }) {
  const [activeCitation, setActiveCitation] = useState(null);

  // messageData matches the unified ChatResponse JSON contract
  const { text, citations, routed_agent } = messageData;

  return (
    <div className="flex flex-col gap-3 w-full text-slate-200">
      {/* Primary Message */}
      <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 leading-relaxed">
        {text}
      </div>

      {/* Render Citation Chips for RAG responses */}
      {routed_agent === 'rag_agent' && citations?.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-1">
          {citations.map((cite, idx) => (
            <button
              key={idx}
              onClick={() => setActiveCitation(cite)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-700 hover:border-cyan-500 hover:text-cyan-400 transition-colors text-xs font-medium"
            >
              <Target className="w-3.5 h-3.5" />
              View Source [{cite.citation_index}]
            </button>
          ))}
        </div>
      )}

      {/* Evidence Viewer Overlay / Split-Screen */}
      {activeCitation && (
        <div className="mt-4 h-[500px] animate-in fade-in slide-in-from-top-4">
          <div className="flex justify-end mb-2">
            <button 
              onClick={() => setActiveCitation(null)}
              className="text-xs text-slate-400 hover:text-red-400 transition-colors"
            >
              Close Viewer
            </button>
          </div>
          <CitationViewer 
            documentUrl={activeCitation.document_url}
            pageNumber={activeCitation.page_number}
            bbox={activeCitation.bbox}
          />
        </div>
      )}
    </div>
  );
}
