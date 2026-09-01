import React, { useState, useEffect } from 'react';
import { 
  FileText, MessageSquare, Users, BookOpen, UploadCloud, Search, 
  CheckSquare, ArrowUpRight, Sparkles, Activity, ShieldCheck, 
  Briefcase, RefreshCw
} from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { ErrorState } from '../components/ui/ErrorState';


export const Dashboard = () => {
  const { setCurrentPage, authUser, userMode } = useAppContext();
  const [stats, setStats] = useState({ documents: 0, queries: 0, users: 0, courses: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    const { ok, data, error: err } = await callBackend('get', '/stats/dashboard');
    if (ok && data) {
      setStats({
        documents: data.documents?.total_indexed ?? data.total_documents ?? 0,
        queries: data.queries?.total ?? data.total_queries ?? 0,
        users: (data.users?.students || 0) + (data.users?.teachers || 0) || data.total_users || 0,
        courses: data.courses ?? data.total_courses ?? 0,
      });
    } else {
      // If stats endpoint is unavailable or empty, degrade gracefully
      setError(err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const statCards = [
    {
      title: "Documents Indexed",
      value: stats.documents,
      subtitle: "In multimodal vector store",
      icon: FileText,
      color: "text-blue-600 dark:text-blue-400",
      bg: "bg-blue-50 dark:bg-blue-950/40",
      border: "border-blue-100 dark:border-blue-900/30",
    },
    {
      title: "Queries Processed",
      value: stats.queries,
      subtitle: "RAG & vector answers",
      icon: MessageSquare,
      color: "text-emerald-600 dark:text-emerald-400",
      bg: "bg-emerald-50 dark:bg-emerald-950/40",
      border: "border-emerald-100 dark:border-emerald-900/30",
    },
    {
      title: "Active Learners",
      value: stats.users,
      subtitle: "Students & educators",
      icon: Users,
      color: "text-purple-600 dark:text-purple-400",
      bg: "bg-purple-50 dark:bg-purple-950/40",
      border: "border-purple-100 dark:border-purple-900/30",
    },
    {
      title: "Courses & Curricula",
      value: stats.courses,
      subtitle: "NCERT & specialized paths",
      icon: BookOpen,
      color: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-50 dark:bg-amber-950/40",
      border: "border-amber-100 dark:border-amber-900/30",
    }
  ];

  const quickShortcuts = [
    {
      id: 'upload',
      title: 'Ingest Materials',
      desc: 'Upload PDFs, textbooks, notes, or web articles to index.',
      icon: UploadCloud,
      color: 'from-blue-600 to-indigo-600',
    },
    {
      id: 'search',
      title: 'Search & Chat',
      desc: 'Query your indexed library with citations and follow-up ideas.',
      icon: Search,
      color: 'from-indigo-600 to-purple-600',
    },
    {
      id: 'quiz',
      title: 'Adaptive Quizzes',
      desc: 'Test your understanding with dynamically generated MCQ exams.',
      icon: CheckSquare,
      color: 'from-purple-600 to-pink-600',
    },
    {
      id: 'counselor',
      title: 'AI Career Mentor',
      desc: 'Get personalized academic roadmap guidance and advice.',
      icon: Briefcase,
      color: 'from-amber-500 to-rose-500',
    }
  ];

  return (
    <div className="space-y-8 animate-fade-in text-left">
      {/* Hero Greeting Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-indigo-700 via-indigo-600 to-indigo-900 text-white p-8 sm:p-10 shadow-lg shadow-indigo-950/10">
        {/* Decorative background glow */}
        <div className="absolute -top-24 -right-24 w-80 h-80 bg-white/10 rounded-full blur-2xl pointer-events-none" />
        <div className="absolute -bottom-20 -left-20 w-60 h-60 bg-indigo-500/20 rounded-full blur-xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/15 backdrop-blur-md text-xs font-semibold text-indigo-100">
            <Sparkles size={14} className="text-amber-300" />
            <span>Active Persona: <span className="capitalize font-bold text-white">{userMode}</span></span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            Welcome back, {authUser || 'Learner'}! 👋
          </h1>

          <p className="text-sm sm:text-base text-indigo-100/90 leading-relaxed max-w-2xl">
            Sahayak AI is ready. You can ingest study documents, ask grounded RAG questions, practice adaptive quizzes, or explore curriculum roadmaps.
          </p>

          <div className="flex flex-wrap gap-3 pt-2">
            <Button
              onClick={() => setCurrentPage('upload')}
              className="bg-white text-indigo-900 hover:bg-indigo-50 shadow-md font-semibold"
              size="md"
              icon={UploadCloud}
            >
              Upload Material
            </Button>
            <Button
              onClick={() => setCurrentPage('search')}
              variant="outline"
              className="border-white/30 text-white hover:bg-white/15"
              size="md"
              icon={Search}
            >
              Start Q&A Chat
            </Button>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Activity size={18} className="text-indigo-600 dark:text-indigo-400" />
            Knowledge Base Metrics
          </h2>
          <Button
            size="sm"
            variant="ghost"
            icon={RefreshCw}
            loading={loading}
            onClick={fetchStats}
          >
            Refresh
          </Button>
        </div>

        {error && (
          <div className="mb-4">
            <ErrorState
              title="Could not load real-time statistics"
              error={error}
              onRetry={fetchStats}
            />
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {statCards.map((stat, i) => {
            const Icon = stat.icon;
            return (
              <Card key={i} className={`flex items-start gap-4 ${stat.border} hover:shadow-card-hover transition-all`}>
                <div className={`p-3 rounded-2xl ${stat.bg} ${stat.color} shrink-0`}>
                  <Icon size={24} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 truncate">
                    {stat.title}
                  </p>
                  <h3 className="text-2xl font-black text-slate-900 dark:text-white mt-1">
                    {loading ? (
                      <span className="inline-block w-16 h-7 bg-slate-200 dark:bg-slate-800 rounded animate-pulse" />
                    ) : (
                      stat.value.toLocaleString()
                    )}
                  </h3>
                  <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
                    {stat.subtitle}
                  </p>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Quick Access Module Hub */}
      <div>
        <h2 className="text-base font-bold text-slate-900 dark:text-white mb-4">
          Quick Workspaces
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {quickShortcuts.map((item) => {
            const Icon = item.icon;
            return (
              <Card
                key={item.id}
                hoverable
                onClick={() => setCurrentPage(item.id)}
                className="group flex items-start gap-4 p-5 hover:border-indigo-200 dark:hover:border-indigo-800"
              >
                <div className={`p-3.5 rounded-2xl bg-gradient-to-tr ${item.color} text-white shadow-md shrink-0 group-hover:scale-105 transition-transform`}>
                  <Icon size={22} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                      {item.title}
                    </h3>
                    <ArrowUpRight size={16} className="text-slate-400 group-hover:text-indigo-600 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Knowledge & System Status footer banner */}
      <Card className="bg-slate-50/80 dark:bg-slate-900/40 border-dashed border-slate-200 dark:border-slate-800 p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3 text-left">
          <div className="p-2 rounded-xl bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400">
            <ShieldCheck size={20} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-900 dark:text-white">Sahayak Vector & RAG Engine Active</h4>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">Multimodal ingestion, Qdrant vector database, and Groq/Gemini models connected.</p>
          </div>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setCurrentPage('settings')}
        >
          Check Connectivity
        </Button>
      </Card>
    </div>
  );
};
