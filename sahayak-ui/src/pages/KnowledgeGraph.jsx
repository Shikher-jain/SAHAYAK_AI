import React, { useState, useEffect } from 'react';
import { 
  Share2, Search, RefreshCw 
} from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { ErrorState } from '../components/ui/ErrorState';

export const KnowledgeGraph = () => {
  const { setCurrentPage } = useAppContext();
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });

  const [selectedEntity, setSelectedEntity] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchGraph = async () => {
    setLoading(true);
    setError(null);

    const { ok, data, error: err } = await callBackend('get', '/knowledge/graph');
    if (ok && data) {
      const nodes = data.nodes || data.entities || [];
      const edges = data.edges || data.relationships || [];
      setGraphData({ nodes, edges });
      if (nodes.length > 0) setSelectedEntity(nodes[0]);
    } else {
      setError(err);
      // Fallback sample knowledge graph nodes & relations
      const sampleNodes = [
        { id: 'photosynthesis', label: 'Photosynthesis', type: 'Biological Process', description: 'Process by which green plants synthesize nutrients using sunlight and chlorophyll.' },
        { id: 'chlorophyll', label: 'Chlorophyll', type: 'Pigment Molecule', description: 'Green pigment responsible for absorption of light to provide energy for photosynthesis.' },
        { id: 'chloroplast', label: 'Chloroplast', type: 'Cell Organelle', description: 'Plastid containing chlorophyll in which photosynthesis takes place.' },
        { id: 'glucose', label: 'Glucose (C6H12O6)', type: 'Chemical Compound', description: 'Simple sugar that is an important energy source in living organisms.' },
        { id: 'carbon_dioxide', label: 'Carbon Dioxide (CO2)', type: 'Reactant Gas', description: 'Atmospheric gas required by plants for the Calvin cycle.' },
        { id: 'light_reaction', label: 'Light-dependent Reaction', type: 'Pathway Phase', description: 'Phase occurring in thylakoid membranes generating ATP and NADPH.' },
      ];
      const sampleEdges = [
        { source: 'photosynthesis', target: 'chlorophyll', relation: 'uses' },
        { source: 'chlorophyll', target: 'chloroplast', relation: 'located_in' },
        { source: 'photosynthesis', target: 'glucose', relation: 'produces' },
        { source: 'photosynthesis', target: 'carbon_dioxide', relation: 'consumes' },
        { source: 'photosynthesis', target: 'light_reaction', relation: 'contains_phase' },
      ];
      setGraphData({ nodes: sampleNodes, edges: sampleEdges });
      setSelectedEntity(sampleNodes[0]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchGraph();
  }, []);

  const filteredNodes = (graphData.nodes || []).filter((n) => {
    const label = (typeof n === 'object' ? n.label || n.name || n.id : n).toLowerCase();
    return label.includes(searchQuery.toLowerCase());
  });

  const getRelationsForEntity = (entityId) => {
    return (graphData.edges || []).filter(
      (e) => e.source === entityId || e.target === entityId
    );
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Knowledge Graph Explorer"
        subtitle="Explore semantic connections, conceptual dependencies, and relational entities extracted across your indexed documents."
        badge={<Badge variant="primary" size="md">Graph RAG</Badge>}
        action={
          <Button
            size="sm"
            variant="ghost"
            icon={RefreshCw}
            loading={loading}
            onClick={fetchGraph}
          >
            Refresh
          </Button>
        }
      />

      {error && (
        <ErrorState
          title="Could not connect to knowledge graph engine"
          error={error}
          onRetry={fetchGraph}
        />
      )}

      {/* Main Graph & Entity Detail layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Entity Node List */}
        <div className="lg:col-span-5 space-y-4">
          <Input
            placeholder="Filter entities by name..."
            icon={Search}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          <Card className="p-4 max-h-[500px] overflow-y-auto space-y-1.5">
            {loading ? (
              <Skeleton variant="text" count={6} />
            ) : filteredNodes.length === 0 ? (
              <p className="text-xs text-slate-400 p-4 text-center">No matching entities found.</p>
            ) : (
              filteredNodes.map((node, idx) => {
                const id = typeof node === 'object' ? node.id || node.name : node;
                const label = typeof node === 'object' ? node.label || node.name || node.id : node;
                const type = typeof node === 'object' ? node.type : 'Concept';
                const isSelected = selectedEntity && (selectedEntity.id === id || selectedEntity === node);

                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setSelectedEntity(node)}
                    className={`
                      w-full p-3 rounded-xl text-left text-xs transition-all flex items-center justify-between
                      ${isSelected 
                        ? 'bg-indigo-50 dark:bg-indigo-950/70 border border-indigo-200 dark:border-indigo-800/80 text-indigo-950 dark:text-indigo-200 shadow-xs' 
                        : 'bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 border border-transparent'}
                    `}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className={`w-2.5 h-2.5 rounded-full ${isSelected ? 'bg-indigo-600' : 'bg-slate-300 dark:bg-slate-700'}`} />
                      <span className="font-semibold truncate">{label}</span>
                    </div>
                    {type && (
                      <span className="text-[10px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-500 shrink-0 ml-2">
                        {type}
                      </span>
                    )}
                  </button>
                );
              })
            )}
          </Card>
        </div>

        {/* Right Column: Selected Entity Deep Dive & Relations */}
        <div className="lg:col-span-7 space-y-6">
          {selectedEntity ? (
            <Card className="p-6 space-y-6">
              <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800">
                <div>
                  <Badge variant="primary" size="sm" className="mb-2">
                    {selectedEntity.type || 'Entity Node'}
                  </Badge>
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                    {selectedEntity.label || selectedEntity.name || selectedEntity.id || selectedEntity}
                  </h3>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setCurrentPage('search')}
                >
                  Ask AI
                </Button>
              </div>

              {selectedEntity.description && (
                <div className="space-y-1">
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                    Description & Concept Definition:
                  </h4>
                  <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed bg-slate-50 dark:bg-slate-950/40 p-4 rounded-xl border border-slate-200/60 dark:border-slate-800/60">
                    {selectedEntity.description}
                  </p>
                </div>
              )}

              {/* Connected Relationships */}
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                  Connected Graph Relationships:
                </h4>
                <div className="space-y-2">
                  {getRelationsForEntity(selectedEntity.id || selectedEntity.label).length === 0 ? (
                    <p className="text-xs text-slate-400">No active relationships recorded for this node.</p>
                  ) : (
                    getRelationsForEntity(selectedEntity.id || selectedEntity.label).map((edge, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-slate-50 dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 flex items-center justify-between text-xs font-mono"
                      >
                        <span className="font-bold text-slate-900 dark:text-white">{edge.source}</span>
                        <span className="px-2 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 text-[11px] font-sans font-semibold">
                          — {edge.relation || 'relates_to'} →
                        </span>
                        <span className="font-bold text-slate-900 dark:text-white">{edge.target}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </Card>
          ) : (
            <Card className="p-12 text-center text-slate-400">
              <Share2 size={36} className="mx-auto mb-3 opacity-50" />
              <p className="text-xs">Select any entity node on the left to inspect its graph connections.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
