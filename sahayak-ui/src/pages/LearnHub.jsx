import React, { useState } from 'react';
import { 
  Search, Plus, Trash2 
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';

export const LearnHub = () => {
  const { showSuccess, setCurrentPage } = useAppContext();
  const [searchQuery, setSearchQuery] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newNote, setNewNote] = useState({ title: '', subject: 'Science', body: '' });

  const [notes, setNotes] = useState(() => {
    const saved = localStorage.getItem('sahayak_study_notes');
    if (saved) {
      try { return JSON.parse(saved); } catch (e) {}
    }
    return [
      {
        id: '1',
        title: 'Key Thermodynamics Laws & Formulas',
        subject: 'Physics',
        body: 'First Law: ΔU = Q - W. Second Law: Entropy of an isolated system always increases. Carnot Efficiency: η = 1 - (Tc/Th).',
        date: 'Today'
      },
      {
        id: '2',
        title: 'Photosynthesis Light vs Dark Reaction Summary',
        subject: 'Biology',
        body: 'Light reaction takes place in Thylakoids (produces ATP & NADPH). Calvin Cycle (Dark Reaction) takes place in Stroma (fixes CO2 into Glucose).',
        date: 'Yesterday'
      },
      {
        id: '3',
        title: 'Fundamental Rights in Indian Constitution',
        subject: 'Civics',
        body: 'Articles 12-35 in Part III: Right to Equality (14-18), Right to Freedom (19-22), Right against Exploitation (23-24), Freedom of Religion (25-28).',
        date: '3 days ago'
      }
    ];
  });

  const saveNotesToStorage = (updated) => {
    setNotes(updated);
    localStorage.setItem('sahayak_study_notes', JSON.stringify(updated));
  };

  const handleAddNote = (e) => {
    e.preventDefault();
    if (!newNote.title.trim() || !newNote.body.trim()) return;

    const item = {
      id: String(Date.now()),
      title: newNote.title.trim(),
      subject: newNote.subject,
      body: newNote.body.trim(),
      date: 'Just now'
    };

    const updated = [item, ...notes];
    saveNotesToStorage(updated);
    showSuccess('Study note added!');
    setIsModalOpen(false);
    setNewNote({ title: '', subject: 'Science', body: '' });
  };

  const handleDeleteNote = (id) => {
    const updated = notes.filter(n => n.id !== id);
    saveNotesToStorage(updated);
    showSuccess('Note deleted.');
  };

  const filteredNotes = notes.filter(n =>
    n.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    n.body.toLowerCase().includes(searchQuery.toLowerCase()) ||
    n.subject.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Personal Study Hub"
        subtitle="Review your saved AI explanations, summary cards, formula sheets, and study notes."
        badge={<Badge variant="primary" size="md">Study Notes</Badge>}
        action={
          <Button
            size="sm"
            icon={Plus}
            onClick={() => setIsModalOpen(true)}
          >
            New Study Card
          </Button>
        }
      />

      {/* Quick Search */}
      <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
        <div className="w-full sm:w-80">
          <Input
            placeholder="Search study cards..."
            icon={Search}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Grid of Study Notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredNotes.map((note) => (
          <Card
            key={note.id}
            className="flex flex-col justify-between p-5 space-y-4 hover:border-indigo-300 dark:hover:border-indigo-700 transition-all shadow-sm"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Badge variant="primary" size="sm">{note.subject}</Badge>
                <span className="text-[10px] text-slate-400">{note.date}</span>
              </div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white leading-snug">
                {note.title}
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans whitespace-pre-wrap">
                {note.body}
              </p>
            </div>

            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <Button
                size="sm"
                variant="ghost"
                className="text-xs text-indigo-600 dark:text-indigo-400 p-0"
                onClick={() => setCurrentPage('search')}
              >
                Ask AI Follow-up →
              </Button>
              <button
                onClick={() => handleDeleteNote(note.id)}
                className="text-slate-400 hover:text-rose-500 p-1 rounded transition-colors"
                title="Delete note"
              >
                <Trash2 size={14} />
              </button>
            </div>
          </Card>
        ))}
      </div>

      {/* Add Study Card Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Create a New Study Note Card"
        subtitle="Save formulas, concepts, or synthesized definitions."
      >
        <form onSubmit={handleAddNote} className="space-y-4 text-left">
          <Input
            label="Concept / Formula Title"
            placeholder="e.g. Kirchhoff's Current & Voltage Laws"
            required
            value={newNote.title}
            onChange={(e) => setNewNote({ ...newNote, title: e.target.value })}
          />

          <Input
            label="Subject"
            placeholder="e.g. Physics / Mathematics / Chemistry"
            value={newNote.subject}
            onChange={(e) => setNewNote({ ...newNote, subject: e.target.value })}
          />

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Note Details & Formulas
            </label>
            <textarea
              rows={4}
              required
              placeholder="Write summary notes, key steps, or formulas..."
              value={newNote.body}
              onChange={(e) => setNewNote({ ...newNote, body: e.target.value })}
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
              Save Study Card
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
};
