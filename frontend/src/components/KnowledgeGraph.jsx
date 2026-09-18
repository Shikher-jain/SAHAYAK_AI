import React, { useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

export default function KnowledgeGraph({ graphData }) {
  const fgRef = useRef();

  const handleNodeClick = useCallback(node => {
    // Center/zoom on node when clicked
    fgRef.current.centerAt(node.x, node.y, 1000);
    fgRef.current.zoom(8, 2000);
  }, [fgRef]);

  return (
    <div className="w-full h-full bg-slate-950">
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="id"
        nodeAutoColorBy="group"
        onNodeClick={handleNodeClick}
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={d => d.value * 0.001}
      />
    </div>
  );
}
