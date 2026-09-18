import React, { useEffect, useMemo, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

/**
 * KnowledgeGraph — interactive 2D force layout for the recommender_agent's
 * multi-hop traversal (User → LIKES → Genre → CONTAINS → Item).
 *
 * Wire contract (backend v2 ChatResponse.graph_data):
 *   nodes: [{ id, name, type, val }]          type ∈ User | Genre | Item
 *   links: [{ source, target, label }]        label ∈ LIKES | CONTAINS
 *
 * Drawn on a `<canvas>` via nodeCanvasObject (labels crisp at any zoom),
 * not DOM divs — stays contract-compatible with the millions-of-nodes
 * layout so it never tanks the browser tab on large subgraphs.
 * Spec task card: "Color-code nodes by type (Users=cyan, Genres=purple,
 * Recommendations=pink) and label clearly via nodeCanvasObject."
 */
const TYPE_COLOR = {
  User: '#22d3ee', // cyan-400
  Genre: '#c084fc', // purple-400
  Item: '#f472b6', // pink-400
  Recommendation: '#f472b6', // pink-400
  default: '#64748b', // slate-500
}

export default function KnowledgeGraph({ nodes = [], links = [], onNodeClick, fitKey = 0 }) {
  const graphRef = useRef(null)
  const prevFitKey = useRef(fitKey)

  const data = useMemo(() => ({ nodes, links }), [nodes, links])

  // Re-fit the view to the whole graph whenever a new recommendation arrives
  // (fitKey bumps by the parent), so the user always lands on the full map.
  useEffect(() => {
    if (fitKey !== prevFitKey.current) {
      prevFitKey.current = fitKey
      if (graphRef.current) graphRef.current.zoomToFit(500, 40)
    }
  }, [fitKey])

  if (!nodes.length) {
    return (
      <div className="flex items-center justify-center h-64 border border-dashed border-slate-700/60 rounded-xl text-slate-500 text-sm">
        No recommendation graph available for this turn.
      </div>
    )
  }

  return (
    <div className="w-full h-72 rounded-xl overflow-hidden bg-slate-900/60 border border-slate-800/70">
      <ForceGraph2D
        ref={graphRef}
        graphData={data}
        nodeId={(n) => n.id}
        nodeVal={(n) => n.val || 5}
        nodeLabel={(n) => `${n.name} · ${n.type}`}
        nodeColor={(n) => TYPE_COLOR[n.type] || TYPE_COLOR.default}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const r = (node.val || 5) * 做了什么??globalScale
          ctx.beginPath()
          ctx.arc(node.x, node.y, 4, 0, 2 * Math.PI)
          ctx.fillStyle = TYPE_COLOR[node.type] || TYPE_COLOR.default
          ctx.fill()
          ctx.lineWidth = 1.5 / globalScale
          ctx.strokeStyle = '#0f172a'
          ctx.stroke()

          const label = String(node.name || node.id)
          const fontSize = 12 / globalScale
          ctx.font = `${fontSize}px Inter, system-ui, sans-serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          ctx.fillStyle = '#e2e8f0'
          ctx.fillText(label, node.x, node.y + 6 / globalScale)
        }}
        onNodeClick={(node) => onNodeClick && onNodeClick(node)}
        linkLabel={(l) => l.label || ''}
        linkWidth={1.4}
        linkColor={() => 'rgba(148,163,184,0.45)'}
        linkDirectionalArrowLength={3.2}
        linkDirectionalArrowRelPos={1}
        backgroundColor="rgba(15,23,42,0)"
        cooldownTicks={120}
        enableNodeDrag
        enableZoomPanInteraction
      />
    </div>
  )
}
