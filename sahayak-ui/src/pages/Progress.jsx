import React, { useState, useEffect } from 'react';
import { 
  Clock, Award, BookOpen, CheckCircle, 
  RefreshCw, Plus, Edit2 
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { ErrorState } from '../components/ui/ErrorState';
import { Modal } from '../components/ui/Modal';

export const Progress = () => {
  const { showSuccess, setCurrentPage } = useAppContext();
  const [data, setData] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editingCourse, setEditingCourse] = useState(null);
  const [newPercentage, setNewPercentage] = useState(50);

  const fetchProgress = async () => {
    setLoading(true);
    setError(null);
    const { ok, data: resData, error: err } = await callBackend('get', '/progress');
    if (ok && resData) {
      setData(resData);
    } else {
      setError(err);
      // Fallback sample progress metrics
      setData({
        total_courses: 4,
        completed_courses: 1,
        total_time_minutes: 320,
        average_quiz_score: 84.5,
        courses: [
          {
            course_id: "NCERT Class 10 Science",
            title: "Physics & Chemistry Foundations",
            completion_percentage: 75,
            last_studied: "Today"
          },
          {
            course_id: "Calculus & Linear Algebra",
            title: "Mathematics Core",
            completion_percentage: 45,
            last_studied: "Yesterday"
          },
          {
            course_id: "Python AI & LLM Engineering",
            title: "Applied Computer Science",
            completion_percentage: 90,
            last_studied: "3 days ago"
          },
          {
            course_id: "Indian Constitutional Framework",
            title: "Civics & Governance",
            completion_percentage: 100,
            last_studied: "Last week"
          }
        ]
      });
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchProgress();
  }, []);

  const handleUpdateCourse = async () => {
    if (!editingCourse) return;
    await callBackend('put', `/progress/${encodeURIComponent(editingCourse.course_id)}`, {
      completion_percentage: Number(newPercentage)
    });

    
    // Optimistic update
    setData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        courses: (prev.courses || []).map((c) =>
          c.course_id === editingCourse.course_id
            ? { ...c, completion_percentage: Number(newPercentage) }
            : c
        )
      };
    });

    showSuccess(`Updated progress for ${editingCourse.course_id}!`);
    setEditingCourse(null);
  };

  const statBoxes = [
    {
      label: "Enrolled Courses",
      value: data?.total_courses || (data?.courses?.length ?? 0),
      icon: BookOpen,
      color: "text-blue-600 dark:text-blue-400",
      bg: "bg-blue-50 dark:bg-blue-950/40"
    },
    {
      label: "Completed Modules",
      value: data?.completed_courses || (data?.courses?.filter(c => c.completion_percentage >= 100).length ?? 0),
      icon: CheckCircle,
      color: "text-emerald-600 dark:text-emerald-400",
      bg: "bg-emerald-50 dark:bg-emerald-950/40"
    },
    {
      label: "Study Time",
      value: `${Math.floor((data?.total_time_minutes || 0) / 60)}h ${(data?.total_time_minutes || 0) % 60}m`,
      icon: Clock,
      color: "text-purple-600 dark:text-purple-400",
      bg: "bg-purple-50 dark:bg-purple-950/40"
    },
    {
      label: "Average Quiz Score",
      value: `${(data?.average_quiz_score || 0).toFixed(0)}%`,
      icon: Award,
      color: "text-amber-600 dark:text-amber-400",
      bg: "bg-amber-50 dark:bg-amber-950/40"
    }
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Learning Progress & Analytics"
        subtitle="Monitor course completion rates, quiz scores, and continuous study momentum."
        badge={<Badge variant="primary" size="md">Analytics</Badge>}
        action={
          <Button
            size="sm"
            variant="ghost"
            icon={RefreshCw}
            loading={loading}
            onClick={fetchProgress}
          >
            Refresh
          </Button>
        }
      />

      {error && (
        <ErrorState
          title="Could not connect to progress telemetry"
          error={error}
          onRetry={fetchProgress}
        />
      )}

      {/* Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {statBoxes.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <Card key={i} className="p-5 flex items-start gap-4">
              <div className={`p-3 rounded-2xl ${stat.bg} ${stat.color} shrink-0`}>
                <Icon size={22} />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 truncate">
                  {stat.label}
                </p>
                <h3 className="text-2xl font-black text-slate-900 dark:text-white mt-1">
                  {loading ? <Skeleton variant="title" className="w-12 h-6" /> : stat.value}
                </h3>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Active Courses Progress Bar List */}
      <Card className="p-6 sm:p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900 dark:text-white">
              Course Completion Trackers
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Keep progressing through syllabus chapters and quizzes.
            </p>
          </div>
          <Button
            size="sm"
            icon={Plus}
            onClick={() => setCurrentPage('books')}
          >
            Explore More Courses
          </Button>
        </div>

        <div className="space-y-6 pt-2">
          {(data?.courses || []).map((course, idx) => {
            const pct = Math.min(100, Math.max(0, course.completion_percentage || 0));
            const isFinished = pct >= 100;

            return (
              <div
                key={idx}
                className="p-4 rounded-2xl bg-slate-50/70 dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 space-y-3"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                      {course.title || course.course_id}
                    </h3>
                    <p className="text-xs text-slate-400 font-mono">
                      ID: {course.course_id} {course.last_studied && `• Last studied: ${course.last_studied}`}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Badge variant={isFinished ? 'success' : 'primary'} size="sm">
                      {pct}%
                    </Badge>
                    <button
                      onClick={() => {
                        setEditingCourse(course);
                        setNewPercentage(pct);
                      }}
                      className="text-slate-400 hover:text-indigo-600 p-1 rounded-md"
                      title="Update percentage"
                    >
                      <Edit2 size={14} />
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-slate-200/80 dark:bg-slate-800 h-2.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${isFinished ? 'bg-emerald-500' : 'bg-indigo-600'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Edit Progress Modal */}
      {editingCourse && (
        <Modal
          isOpen={Boolean(editingCourse)}
          onClose={() => setEditingCourse(null)}
          title={`Update Progress: ${editingCourse.title || editingCourse.course_id}`}
          actions={
            <>
              <Button variant="outline" size="sm" onClick={() => setEditingCourse(null)}>
                Cancel
              </Button>
              <Button size="sm" onClick={handleUpdateCourse}>
                Save Progress
              </Button>
            </>
          }
        >
          <div className="space-y-4 text-xs text-left">
            <label className="font-semibold text-slate-700 dark:text-slate-300 block">
              Completion Percentage (0 - 100%):
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="100"
                value={newPercentage}
                onChange={(e) => setNewPercentage(e.target.value)}
                className="flex-1 accent-indigo-600"
              />
              <span className="font-bold text-sm text-indigo-600 w-12 text-right">
                {newPercentage}%
              </span>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
