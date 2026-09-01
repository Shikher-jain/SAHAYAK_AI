import React, { useState, useEffect } from 'react';
import { 
  Map, ExternalLink, CheckCircle, Circle, Search, 
  BookOpen, RefreshCw 
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { EmptyState } from '../components/ui/EmptyState';
import { ErrorState } from '../components/ui/ErrorState';

export const Roadmaps = () => {
  const { showSuccess, setCurrentPage } = useAppContext();

  const [roadmaps, setRoadmaps] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [progressMap, setProgressMap] = useState({});

  const fetchRoadmaps = async () => {
    setLoading(true);
    setError(null);
    const { ok, data, error: err } = await callBackend('get', '/roadmaps');
    if (ok && data) {
      const items = Array.isArray(data) ? data : (data.roadmaps || []);
      setRoadmaps(items);
    } else {
      setError(err);
      // Fallback sample roadmap data if backend returns empty
      setRoadmaps([
        {
          id: 'frontend',
          title: 'Frontend Web Developer',
          description: 'Step by step guide to becoming a modern frontend developer in 2026 (HTML, CSS, JS, React, Tailwind).',
          difficulty: 'Beginner to Intermediate',
          url: 'https://roadmap.sh/frontend',
          topics: ['HTML & Semantic Web', 'CSS & Flexbox/Grid', 'JavaScript ES6+', 'React Framework', 'Vite & State Mgmt']
        },
        {
          id: 'backend',
          title: 'Backend & API Engineer',
          description: 'Master server-side architecture, FastAPI, relational databases, vector search, and authentication.',
          difficulty: 'Intermediate to Advanced',
          url: 'https://roadmap.sh/backend',
          topics: ['Python & FastAPI', 'PostgreSQL & Qdrant', 'JWT Auth & Security', 'Docker & Deployment', 'Microservices']
        },
        {
          id: 'ai-engineer',
          title: 'AI & Multimodal RAG Specialist',
          description: 'Learn LLM prompting, embeddings, vector indexing, LangChain, semantic retrieval, and fine-tuning.',
          difficulty: 'Advanced',
          url: 'https://roadmap.sh/ai-data-scientist',
          topics: ['Vector Databases', 'Prompt Engineering', 'RAG Architectures', 'Multimodal Models', 'Agents & Tool Calling']
        },
        {
          id: 'devops',
          title: 'DevOps & Cloud Infrastructure',
          description: 'Guide to CI/CD pipelines, Docker containers, Kubernetes clusters, and cloud computing.',
          difficulty: 'Intermediate',
          url: 'https://roadmap.sh/devops',
          topics: ['Linux & Shell', 'Docker & Compose', 'CI/CD GitHub Actions', 'Kubernetes', 'Cloud Security']
        }
      ]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchRoadmaps();
  }, []);

  const toggleTopicProgress = (roadmapId, topic) => {
    setProgressMap((prev) => {
      const current = prev[roadmapId] || [];
      const updated = current.includes(topic)
        ? current.filter(t => t !== topic)
        : [...current, topic];
      return { ...prev, [roadmapId]: updated };
    });
    showSuccess('Progress saved!');
  };

  const filteredRoadmaps = roadmaps.filter((r) => 
    r.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    r.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Learning Roadmaps"
        subtitle="Structured, milestone-driven paths to master new engineering, design, and academic disciplines."
        badge={<Badge variant="primary" size="md">Curricula</Badge>}
        action={
          <Button
            size="sm"
            variant="ghost"
            icon={RefreshCw}
            loading={loading}
            onClick={fetchRoadmaps}
          >
            Refresh
          </Button>
        }
      />

      {/* Search & Filter Bar */}
      <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
        <div className="w-full sm:w-80">
          <Input
            placeholder="Search roadmaps by title or skill..."
            icon={Search}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {error && (
        <ErrorState
          title="Could not connect to roadmaps service"
          error={error}
          onRetry={fetchRoadmaps}
        />
      )}

      {/* Loading Skeletons */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton variant="card" count={4} />
        </div>
      )}

      {/* Empty State */}
      {!loading && filteredRoadmaps.length === 0 && (
        <EmptyState
          icon={Map}
          title="No Roadmaps Found"
          description="Try modifying your search query or clear the filter."
          actionLabel="Clear Search"
          onAction={() => setSearchQuery('')}
        />
      )}

      {/* Roadmaps Grid */}
      {!loading && filteredRoadmaps.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {filteredRoadmaps.map((rm) => {
            const completedTopics = progressMap[rm.id] || [];
            const topics = rm.topics || ['Foundations', 'Core Concepts', 'Tooling', 'Advanced Projects'];
            const percent = Math.round((completedTopics.length / topics.length) * 100);

            return (
              <Card
                key={rm.id || rm.title}
                className="flex flex-col justify-between p-6 space-y-5 hover:border-indigo-300 dark:hover:border-indigo-700 transition-all shadow-sm"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="p-2.5 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 shrink-0">
                      <Map size={22} />
                    </div>
                    {rm.difficulty && (
                      <Badge variant="neutral" size="sm">
                        {rm.difficulty}
                      </Badge>
                    )}
                  </div>

                  <div>
                    <h3 className="text-base font-bold text-slate-900 dark:text-white">
                      {rm.title}
                    </h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed line-clamp-2">
                      {rm.description}
                    </p>
                  </div>

                  {/* Progress tracker */}
                  <div className="space-y-1.5 pt-1">
                    <div className="flex justify-between text-[11px] font-semibold text-slate-500">
                      <span>Milestones Completed</span>
                      <span className="text-indigo-600 dark:text-indigo-400">{percent}%</span>
                    </div>
                    <div className="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-indigo-600 h-full rounded-full transition-all duration-300"
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                  </div>

                  {/* Milestones list */}
                  <div className="space-y-1.5 pt-2">
                    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      Key Milestones:
                    </p>
                    <div className="space-y-1">
                      {topics.map((t, idx) => {
                        const isDone = completedTopics.includes(t);
                        return (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => toggleTopicProgress(rm.id, t)}
                            className={`w-full flex items-center justify-between p-2 rounded-lg text-xs transition-colors ${isDone ? 'bg-emerald-50/70 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300' : 'bg-slate-50 dark:bg-slate-800/40 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'}`}
                          >
                            <span className="truncate">{t}</span>
                            {isDone ? (
                              <CheckCircle size={14} className="text-emerald-600 dark:text-emerald-400 shrink-0 ml-2" />
                            ) : (
                              <Circle size={14} className="text-slate-300 dark:text-slate-600 shrink-0 ml-2" />
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    icon={BookOpen}
                    onClick={() => setCurrentPage('search')}
                  >
                    Study with AI
                  </Button>

                  {rm.url && (
                    <a
                      href={rm.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline"
                    >
                      <span>Interactive Graph</span>
                      <ExternalLink size={14} />
                    </a>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};
