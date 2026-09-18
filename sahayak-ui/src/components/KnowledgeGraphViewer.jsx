import React, { useRef, useState, useCallback, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { ZoomIn, ZoomOut, Maximize2, RefreshCw, X, Layers } from 'lucide-react';

const NODE_COLORS = {
  Document: '#38bdf8',  // Cyan / Sky blue
  Entity: '#a855f7',    // Purple
  Genre: '#f59e0b',     // Amber
  User: '#10b981',      // Emerald
  default: '#94a3b8'    // Slate
};

export default function KnowledgeGraphViewer({ graphData, onNodeClick }) {
  const fgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [filterGroup, setFilterGroup] = useState('ALL');

  // Filter nodes & links based on active category
  const filteredData = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] };
    if (filterGroup === 'ALL') return graphData;

    const visibleNodeIds = new Set(
      graphData.nodes
        .filter((n) => n.group === filterGroup)
        .map((n) => n.id)
    );

    const filteredNodes = graphData.nodes.filter((n) => visibleNodeIds.has(n.id));
    const filteredLinks = graphData.links.filter(
      (l) => visibleNodeIds.has(l.source.id || l.source) && visibleNodeIds.has(l.target.id || l.target)
    );

    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, filterGroup]);

  // Handle Zoom controls
  const handleZoomIn = () => fgRef.current?.zoom(fgRef.current.zoom() * 1.3, 400);
  const handleZoomOut = () => fgRef.current?.zoom(fgRef.current.zoom() / 1.3, 400);
  const handleZoomReset = () => fgRef.current?.zoomToFit(400, 50);

  // Custom node drawing with canvas
  const drawNode = useCallback((node, ctx, globalScale) => {
    const isSelected = selectedNode?.id === node.id;
    const isHovered = hoveredNode?.id === node.id;
    const radius = isSelected ? 8 : isHovered ? 7 : 5;
    const color = NODE_COLORS[node.group] || NODE_COLORS.default;

    // Glowing halo on hover/select
    if (isSelected || isHovered) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius + 4, 0, 2 * Math.PI, false);
      ctx.fillStyle = isSelected ? 'rgba(56, 189, 248, 0.3)' : 'rgba(255, 255, 255, 0.15)';
      ctx.fill();
    }

    // Core node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();

    // Node border
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.strokeStyle = isSelected ? '#ffffff' : '#0f172a';
    ctx.stroke();

    // Text Label rendering (dynamic scaling)
    if (globalScale >= 1.2 || isSelected || isHovered) {
      const label = node.label || node.id;
      const fontSize = Math.max(10 / globalScale, 3.5);
      ctx.font = `${fontSize}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = isSelected ? '#38bdf8' : '#e2e8f0';
      ctx.fillText(label, node.x, node.y + radius + 2);
    }
  }, [selectedNode, hoveredNode]);

  return (
    <div className="relative w-full h-[650px] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-2xl flex">
      {/* Top Floating Control Bar */}
      <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-slate-900/90 backdrop-blur-md px-3 py-2 rounded-lg border border-slate-800 shadow-md">
        <Layers className="w-4 h-4 text-slate-400" />
        <select
          value={filterGroup}
          onChange={(e) => setFilterGroup(e.target.value)}
          aria-label="Filter entities by category"
          className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
        >
          <option value="ALL" className="bg-slate-900">All Entities</option>
          <option value="Document" className="bg-slate-900">Documents</option>
          <option value="Entity" className="bg-slate-900">Entities</option>
          <option value="Genre" className="bg-slate-900">Genres</option>
          <option value="User" className="bg-slate-900">Users</option>
        </select>
      </div>

      {/* Floating Zoom & Fit Action Buttons */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-1.5 bg-slate-900/90 backdrop-blur-md p-1.5 rounded-lg border border-slate-800 shadow-md">
        <button 
          onClick={handleZoomIn} 
          aria-label="Zoom in graph"
          className="p-1.5 hover:bg-slate-800 rounded text-slate-300 transition"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button 
          onClick={handleZoomOut} 
          aria-label="Zoom out graph"
          className="p-1.5 hover:bg-slate-800 rounded text-slate-300 transition"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button 
          onClick={handleZoomReset} 
          aria-label="Fit graph to view"
          className="p-1.5 hover:bg-slate-800 rounded text-slate-300 transition"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Main Force-Directed Graph Canvas */}
      <div className="flex-1 w-full h-full cursor-grab active:cursor-grabbing">
        <ForceGraph2D
          ref={fgRef}
          graphData={filteredData}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
          }}
          linkColor={() => '#334155'}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.15}
          cooldownTicks={100}
          onNodeClick={(node) => {
            setSelectedNode(node);
            fgRef.current.centerAt(node.x, node.y, 400);
            if (onNodeClick) onNodeClick(node);
          }}
          onNodeHover={setHoveredNode}
        />
      </div>

      {/* Slide-out Metadata Inspector Drawer */}
      {selectedNode && (
        <div className="w-80 bg-slate-900 border-l border-slate-800 p-5 flex flex-col justify-between overflow-y-auto z-20">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs uppercase tracking-wider font-semibold px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
                {selectedNode.group}
              </span>
              <button 
                onClick={() => setSelectedNode(null)} 
                aria-label="Close inspector drawer"
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <h3 className="mt-4 text-base font-semibold text-slate-100 break-words">
              {selectedNode.label || selectedNode.id}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">ID: {selectedNode.id}</p>

            {/* Custom Node Properties */}
            <div className="mt-5 space-y-3">
              <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Metadata</div>
              {selectedNode.properties ? (
                Object.entries(selectedNode.properties).map(([key, value]) => (
                  <div key={key} className="bg-slate-950/60 p-2.5 rounded border border-slate-800/80">
                    <div className="text-[11px] text-slate-400 capitalize">{key}</div>
                    <div className="text-xs text-slate-200 mt-0.5 break-all font-mono">
                      {String(value)}
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-500 italic">No additional properties stored.</p>
              )}
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800">
            <button
              onClick={() => {
                fgRef.current.centerAt(selectedNode.x, selectedNode.y, 400);
                fgRef.current.zoom(2.5, 400);
              }}
              className="w-full py-2 bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/30 rounded text-xs font-medium transition flex items-center justify-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Center & Focus Node
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
