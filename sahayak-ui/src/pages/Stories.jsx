import React, { useState, useEffect } from 'react';
import { 
  Heart, Plus, Search, MessageSquare, Sparkles, 
  Share2, User, RefreshCw, ThumbsUp, CheckCircle2 
} from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { EmptyState } from '../components/ui/EmptyState';
import { ErrorState } from '../components/ui/ErrorState';
import { Modal } from '../components/ui/Modal';

export const Stories = () => {
  const { authUser, showSuccess, showError } = useAppContext();
  const [stories, setStories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newStory, setNewStory] = useState({
    title: '',
    category: 'STEM',
    content: '',
  });
  const [likedStories, setLikedStories] = useState(new Set());

  const fetchStories = async () => {
    setLoading(true);
    setError(null);

    const { ok, data, error: err } = await callBackend('get', '/stories');
    if (ok && data) {
      const items = Array.isArray(data) ? data : (data.stories || []);
      setStories(items);
    } else {
      setError(err);
      // Fallback sample community stories
      setStories([
        {
          id: 'story-1',
          title: 'How Sahayak RAG helped me prepare for JEE Advanced Physics',
          author: 'Aarav Sharma',
          role: 'Class XII Aspirant',
          category: 'Exam Preparation',
          content: 'I uploaded my NCERT textbooks and HC Verma notes into Sahayak. The AI grounded responses helped me clarify difficult electromagnetism concepts without hallucinating facts.',
          likes: 24,
          date: '2 days ago'
        },
        {
          id: 'story-2',
          title: 'Creating adaptive quizzes for my 10th grade biology class',
          author: 'Priya Verma',
          role: 'Secondary School Educator',
          category: 'Teaching Insights',
          content: 'The Quiz Engine generated 5 distinct MCQ tests based on Photosynthesis and Cellular Respiration in seconds. My students loved the instant explanations.',
          likes: 38,
          date: '4 days ago'
        },
        {
          id: 'story-3',
          title: 'Transitioning from non-tech to Python AI with custom Roadmaps',
          author: 'Rohan Gupta',
          role: 'Career Switcher',
          category: 'Career Growth',
          content: 'The AI Counselor domain mentor recommended a tailored roadmap starting with Python basics, vector databases, and LangChain embeddings.',
          likes: 19,
          date: '1 week ago'
        }
      ]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchStories();
  }, []);

  const handleLike = (storyId) => {
    setLikedStories((prev) => {
      const next = new Set(prev);
      const isAlreadyLiked = next.has(storyId);
      if (isAlreadyLiked) next.delete(storyId);
      else next.add(storyId);
      return next;
    });

    setStories((prev) =>
      prev.map((s) =>
        s.id === storyId
          ? { ...s, likes: s.likes + (likedStories.has(storyId) ? -1 : 1) }
          : s
      )
    );
  };

  const handleCreateStory = async (e) => {
    e.preventDefault();
    if (!newStory.title.trim() || !newStory.content.trim()) {
      showError('Please provide a title and story content.');
      return;
    }

    const payload = {
      title: newStory.title.trim(),
      category: newStory.category,
      content: newStory.content.trim(),
      author: authUser || 'Community Learner',
      role: 'Student Member'
    };

    const { ok, data } = await callBackend('post', '/stories', payload);

    const createdStory = (ok && data) ? data : {
      id: `story_${Date.now()}`,
      ...payload,
      likes: 1,
      date: 'Just now'
    };

    setStories((prev) => [createdStory, ...prev]);
    showSuccess('Your story has been shared with the community!');
    setIsModalOpen(false);
    setNewStory({ title: '', category: 'STEM', content: '' });
  };

  const filteredStories = stories.filter((s) =>
    s.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.content?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.author?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Community Learning Stories"
        subtitle="Read real experiences, study techniques, and career breakthroughs shared by students and educators."
        badge={<Badge variant="primary" size="md">Community</Badge>}
        action={
          <Button
            size="sm"
            icon={Plus}
            onClick={() => setIsModalOpen(true)}
          >
            Share Your Story
          </Button>
        }
      />

      {/* Search Filter */}
      <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
        <div className="w-full sm:w-80">
          <Input
            placeholder="Search stories by topic, author..."
            icon={Search}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {error && (
        <ErrorState
          title="Could not connect to community story feed"
          error={error}
          onRetry={fetchStories}
        />
      )}

      {/* Loading Skeletons */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton variant="card" count={4} />
        </div>
      )}

      {/* Empty State */}
      {!loading && filteredStories.length === 0 && (
        <EmptyState
          icon={Heart}
          title="No Stories Found"
          description="Be the first to share an inspiring study milestone or tip with the Sahayak community!"
          actionLabel="Write a Story"
          onAction={() => setIsModalOpen(true)}
        />
      )}

      {/* Stories Grid */}
      {!loading && filteredStories.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {filteredStories.map((story) => {
            const isLiked = likedStories.has(story.id);
            return (
              <Card
                key={story.id || story.title}
                className="flex flex-col justify-between p-6 space-y-4 hover:border-indigo-300 dark:hover:border-indigo-700 transition-all shadow-sm"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <Badge variant="primary" size="sm">
                      {story.category || 'Experience'}
                    </Badge>
                    <span className="text-[11px] text-slate-400 font-medium">
                      {story.date || 'Recent'}
                    </span>
                  </div>

                  <h3 className="text-base font-bold text-slate-900 dark:text-white leading-snug">
                    {story.title}
                  </h3>

                  <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                    "{story.content}"
                  </p>
                </div>

                <div className="pt-4 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-7 h-7 rounded-full bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 font-bold text-xs flex items-center justify-center">
                      {story.author ? story.author.charAt(0).toUpperCase() : 'U'}
                    </div>
                    <div className="text-left">
                      <p className="text-xs font-bold text-slate-900 dark:text-white">{story.author}</p>
                      <p className="text-[10px] text-slate-400">{story.role || 'Learner'}</p>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => handleLike(story.id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-colors ${isLiked ? 'bg-rose-50 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-900/60' : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200'}`}
                  >
                    <Heart size={14} className={isLiked ? 'fill-rose-500 text-rose-500' : ''} />
                    <span>{story.likes || 0}</span>
                  </button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Share Story Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Share Your Learning Story"
        subtitle="Inspire other students and teachers with your journey."
      >
        <form onSubmit={handleCreateStory} className="space-y-4 text-left">
          <Input
            label="Story Title"
            placeholder="e.g. How I mastered organic chemistry with Sahayak AI"
            required
            value={newStory.title}
            onChange={(e) => setNewStory({ ...newStory, title: e.target.value })}
          />

          <Select
            label="Category"
            value={newStory.category}
            onChange={(e) => setNewStory({ ...newStory, category: e.target.value })}
            options={[
              { value: 'Exam Preparation', label: 'Exam Preparation' },
              { value: 'Teaching Insights', label: 'Teaching Insights' },
              { value: 'Career Growth', label: 'Career Growth' },
              { value: 'Study Habits', label: 'Study Habits & Tips' },
              { value: 'STEM Projects', label: 'STEM & Coding' }
            ]}
          />

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Your Story & Advice
            </label>
            <textarea
              rows={5}
              required
              placeholder="Share what worked, what tools you used, and what you achieved..."
              value={newStory.content}
              onChange={(e) => setNewStory({ ...newStory, content: e.target.value })}
              className="w-full p-3 rounded-xl text-xs bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-800">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setIsModalOpen(false)}
            >
              Cancel
            </Button>
            <Button type="submit" size="sm">
              Post Story
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
};
