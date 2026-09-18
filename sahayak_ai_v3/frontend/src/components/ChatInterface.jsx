import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, FileText, Send, TrendingUp, MessageSquare } from 'lucide-react'
import axios from 'axios'
import VoiceRecorder from './VoiceRecorder.jsx'
import KnowledgeGraph from './KnowledgeGraph.jsx'
import CitationViewer from './CitationViewer.jsx'
import { apiBase, scrubText, unmaskText } from '../../api/v2_client.js'

const ROI_COLOR = {
  User: '#22d3ee',
  Genre: '#a855f7',
  Recommendation: '#ec4899',
  Item: '#ec4899',
  default: '#64748b',
}

const EMPTY_GRAPH = { nodes: [], links: [] }

/** Normalize the recommendation payload into {nodes, links} for ForceGraph2D. */
function toGraph(graphData) {
  if (!graphData) return EMPTY_GRAPH
  const rawNodes = graphData.nodes || []
  const rawLinks = graphData.links || []
  if (!rawNodes.length) return EMPTY_GRAPH

  const nodes = rawNodes.map((n, i) => ({
    id: n.id || String(i),
    name: n.name || n.id || `node-${i}`,
    type: n.type || 'Recommendation',
    val: n.val || 6,
  }))
  const links = rawLinks
    .filter(
      (l) =>
        l.source !== null &&
        l.source !== undefined &&
        l.target !== null &&
        l.target !== undefined
    )
    .map((l) => ({
      source: typeof l.source === 'object' ? l.source.id : l.source,
      target: typeof l.target === 'object' ? l.target.id : l.target,
      label: l.label || l.type || '',
    }))
  return { nodes, links }
}

/**
 * ChatInterface — full v2_advanced chat surface.
 *
 *   • text → POST /api/v2/chat/orchestrate (semantic-cache + supervisor)
 *   • voice → MediaRecorder webm → POST /api/v2/chat/voice-stream (Groq
 *             Whisper, async, zero local compute)
 *   • recommendation graph → toggle KnowledgeGraph below the reply,
 *             colored User=cyan / Genre=purple / Rec=¶pink
 *   • RAG citations → `react-pdf` <CitationViewer/> pinned to the exact page/bbox
 *
 * The backend/security PII scrubber runs on the inbound transcript (Mode A
 * keyed to the webhook); `answer` is `final_response` text but it is passed
 * through `unmaskText` right here so the UI shows the caller's own original
 * identifiers, never the internal `__PII_*__` tokens.
 */
export default function ChatInterface() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const scrollRef = useRef(null)

  const append = (m) => setMessages((prev) => [...prev, m])

  const onUserText = async (text) => {
    if (!text.trim() || busy) return
    append({ role: 'user', text })
    setInput('')
    setBusy(true)
    try {
      const { data } = await axios.post(`${apiBase}/chat/orchestrate`, {
        message: text,
        user_id: 'frontend_guest',
        tier: 'free',
      })
      const unmasked = await unmaskText(data?.final_response || data?.final_output || '', data?.pii_map || {})
      append({
        role: 'assistant',
        text: unmasked,
        routed_agent: data?.routed_agent || data?.agent || '',
        graph: toGraph(data?.graph_data),
        citations: data?.citations || [],
        distress: Number(data?.distress_level || 0),
        emergency: Boolean(data?.emergency_flag),
        message_id: data?.message_id,
      })
    } catch (e) {
      append({ role: 'assistant', text: `⚠ Failed to reach the supervisor: ${e?.message || e}`, error: true })
    } finally {
      setBusy(false)
    }
  }

  const onUserVoice = async (transcript) => {
    append({ role: 'user', text: transcript, voice: true })
    await onUserText(transcript)
  }

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  function openCitation(c) {
    return (
      <CitationViewer
        key={`${c.document_url}-${c.page_number}`}
        document_url={c.document_url}
        page_number={c.page_number}
        bbox={c.bbox}
        snippet={c.snippet}
      />
    )
  }

  const agentBadge = (m) =>
    m.routed_agent ? (
      <span className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-cyan-400/90 bg-cyan-400/10 border border-cyan-400/20 px-1.5 py-0.5 rounded">
        <MessageSquare className="w-3 h-3" />
        {m.routed_agent}
      </span>
    ) : null

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto w-full">
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee]" />
          <h1 className="font-semibold tracking-tight text-slate-100">Sahayak AI</h1>
          <span className="text-[11px] text-slate-500">v2_advanced · supervisor</span>
        </div>
        {busy && (
          <span className="text-xs text-cyan-400/80 animate-pulse">thinking…</span>
        )}
      </header>

      {/* ── Message stream ──────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
        {messages.length === 0 && (
          <div className="text-center pt-16 space-y-2">
            <TrendingUp className="w-10 h-10 mx-auto text-cyan-400/40" />
            <p className="text-slate-400 text-sm">
              Ask for a recommendation, paste a favourite genre, or hold the mic and speak.
            </p>
            <p className="text-[11px] text-slate-600">
              Output is grounded in the knowledge graph + cited PDFs — nothing leaves the v2 supervisor unmasked.
            </p>
          </div>
        )}

        {messages.map((m, i) => {
          if (m.role === 'user') {
            return (
              <div key={i} className="flex justify-end">
                <div className="max-w-[78%] bg-cyan-500/10 border border-cyan-500/25 text-slate-100 rounded-2xl rounded-br-sm px-4 py-2.5 text-sm">
                  {m.voice && <span className="mr-1.5 text-cyan-400 text-xs">🎙</span>}
                  {m.text}
                </div>
              </div>
            )
          }
          return (
            <div key={i} className="space-y-3">
              {m.error ? (
                <div className="inline-flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                  <AlertTriangle className="w-4 h-4" />
                  {m.text}
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2">{agentBadge(m)}</div>
                  <div className="max-w-[88%] bg-slate-800/50 border border-slate-700/60 text-slate-100 rounded-2xl rounded-bl-sm px-4 py-3 text-sm whitespace-pre-wrap leading-relaxed">
                    {m.text}
                  </div>
                  {m.distress >= 0.72 && (
                    <div className="flex items-center gap-2 text-amber-400/90 text-xs bg-amber-400/5 border border-amber-400/20 rounded-lg px-3 py-2">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      Distress guard: this response followed the counseling protocol.
                    </div>
                  )}
                  {m.citations?.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {m.citations.map((c, ci) => (
                        <button
                          key={ci}
                          onClick={() => setActiveCitation(openCitation(c))}
                          className="inline-flex items-center gap-1.5 text-xs text-cyan-300 bg-slate-800/70 border border-slate-700 hover:border-cyan-500/50 hover:text-cyan-200 rounded-full pl-2.5 pr-3 py-1 transition-colors"
                        >
                          <FileText className="w-3.5 h-3.5" />
                          {c.document_name || `page ${c.page_number}`} · cit {c.citation_index || ci + 1}
                        </button>
                      ))}
                    </div>
                  )}
                  {m.graph?.nodes?.length > 0 && (
                    <div className="border border-slate-700/60 rounded-xl overflow-hidden bg-slate-900/60">
                      <div className="flex items-center justify-between px-3 py-1.5 border-b border-slate-700/60">
                        <span className="text-[11px] uppercase tracking-wider text-slate-400">
                          Knowledge map · {m.graph.nodes.length} nodes
                        </span>
                        <span className="text-[10px] text-slate-500">
                          User <span className="text-cyan-400">■</span> Genre{' '}
                          <span className="text-purple-400">■</span> Rec{' '}
                          <span className="text-pink-400">■</span>
                        </span>
                      </div>
                      <KnowledgeGraph data={m.graph} />
                    </div>
                  )}
                </>
              )}
            </div>
          )
        })}

        {activeCitation && <div className="mt-2">{activeCitation}</div>}

        {busy && (
          <div className="flex items-center gap-2 text-slate-500 text-sm pl-1">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/70 animate-pulse" />
            supervisor is routing your turn…
          </div>
        )}
        <div ref={scrollRef} />
      </main>

      {/* ── Composer ────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800/80 px-4 py-3">
        <form
          className="flex items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            onUserText(input)
          }}
        >
          <VoiceRecorder
            onTranscribed={onUserVoice}
            disabled={busy}
            apiBase={`${apiBase}/chat`}
          />
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                onUserText(input)
              }
            }}
            rows={1}
            placeholder="Ask Sahayak AI… (Enter to send, Shift+Enter for a new line)"
            className="flex-1 resize-none bg-slate-800/50 border border-slate-700/70 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 outline-none transition-colors"
          />
          <button
            type="submit"
            disabled={!input.trim() || busy}
            className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/20 hover:border-cyan-400/50 disabled:opacity-40 disabled:hover:bg-cyan-500/10 transition-colors"
            title="Send"
          >
            <Send className="w-4.5 h-4.5" />
          </button>
        </form>
      </footer>
    </div>
  )
}
