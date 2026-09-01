import React, { useState, useEffect } from 'react';
import { 
  Lightbulb, Stethoscope, Scale, Palette, 
  Cpu, DollarSign, Globe 
} from 'lucide-react';

import { callBackend } from '../api/client';
import { SearchChat } from './SearchChat';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { PageHeader } from '../components/ui/PageHeader';

export const Counselor = () => {
  const [selectedDomain, setSelectedDomain] = useState('stem');
  const [suggestions, setSuggestions] = useState([]);

  const domains = [
    { id: 'stem', label: 'STEM & Engineering', icon: Cpu, color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-50 dark:bg-blue-950/40', desc: 'Computer Science, AI, Robotics, Physics, Maths' },
    { id: 'medical', label: 'Medical & Healthcare', icon: Stethoscope, color: 'text-rose-600 dark:text-rose-400', bg: 'bg-rose-50 dark:bg-rose-950/40', desc: 'Medicine, Biotech, Nursing, Pharmacy, Research' },
    { id: 'commerce', label: 'Commerce & Finance', icon: DollarSign, color: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-950/40', desc: 'Economics, Accounting, Fintech, Investment, Business' },
    { id: 'arts', label: 'Arts & Humanities', icon: Palette, color: 'text-purple-600 dark:text-purple-400', bg: 'bg-purple-50 dark:bg-purple-950/40', desc: 'Literature, Design, Journalism, History, Psychology' },
    { id: 'law', label: 'Law & Governance', icon: Scale, color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-950/40', desc: 'Corporate Law, Constitutional Rights, Policy, Judiciary' },
    { id: 'general', label: 'General & Interdisciplinary', icon: Globe, color: 'text-indigo-600 dark:text-indigo-400', bg: 'bg-indigo-50 dark:bg-indigo-950/40', desc: 'Career Transitions, Higher Studies, Soft Skills' },
  ];

  useEffect(() => {
    let isMounted = true;
    callBackend('get', `/counselor/suggestions?domain=${selectedDomain}`)
      .then((res) => {
        if (isMounted && res.ok && res.data) {
          setSuggestions(Array.isArray(res.data) ? res.data : (res.data.suggestions || []));
        }
      })
      .catch(() => {});


    return () => { isMounted = false; };
  }, [selectedDomain]);

  return (
    <div className="space-y-6 max-w-6xl mx-auto animate-fade-in text-left">
      <PageHeader
        title="AI Career & Academic Counselor"
        subtitle="Consult specialized AI domain mentors for personalized guidance, curriculum roadmaps, and career planning."
        badge={<Badge variant="primary" size="md">Mentor</Badge>}
      />

      {/* Domain Selection Tabs */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
        {domains.map((d) => {
          const Icon = d.icon;
          const isSelected = selectedDomain === d.id;
          return (
            <button
              key={d.id}
              onClick={() => setSelectedDomain(d.id)}
              className={`
                p-3 rounded-2xl border text-left transition-all duration-150 flex flex-col justify-between
                ${isSelected 
                  ? 'bg-indigo-50/80 dark:bg-indigo-950/50 border-indigo-400 dark:border-indigo-600 shadow-sm' 
                  : 'bg-white dark:bg-slate-900 border-slate-200/80 dark:border-slate-800/80 hover:border-slate-300 dark:hover:border-slate-700'}
              `}
            >
              <div className={`p-2 rounded-xl w-fit ${d.bg} ${d.color} mb-2`}>
                <Icon size={18} />
              </div>
              <div>
                <p className={`text-xs font-bold ${isSelected ? 'text-indigo-950 dark:text-indigo-200' : 'text-slate-900 dark:text-slate-200'}`}>
                  {d.label.split(' ')[0]}
                </p>
                <p className="text-[10px] text-slate-400 line-clamp-1 mt-0.5">
                  {d.label.split('&')[1] || 'Specialization'}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Domain Info & Suggestions */}
      {suggestions.length > 0 && (
        <Card className="p-4 bg-slate-50/70 dark:bg-slate-900/50 border-slate-200/70 dark:border-slate-800/70">
          <div className="flex items-center gap-2 mb-2 text-xs font-bold text-slate-700 dark:text-slate-300">
            <Lightbulb size={16} className="text-amber-500" />
            <span>Recommended Exploration Questions in {selectedDomain.toUpperCase()}:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestions.slice(0, 4).map((s, idx) => (
              <span
                key={idx}
                className="text-xs bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-3 py-1.5 rounded-xl border border-slate-200/60 dark:border-slate-700/60 shadow-2xs"
              >
                {typeof s === 'string' ? s : s.title || s.text || JSON.stringify(s)}
              </span>
            ))}
          </div>
        </Card>
      )}

      {/* Embedded Search/Counselor Chat interface */}
      <SearchChat
        key={selectedDomain}
        endpoint="/counselor/chat"
        title={`Counselor Session: ${domains.find(d => d.id === selectedDomain)?.label}`}
        subtitle="Ask questions regarding course eligibility, entrance exams, job scope, or skill paths."
        placeholder={`Ask your ${selectedDomain.toUpperCase()} career counselor anything...`}
        extraPayload={{ domain: selectedDomain }}
      />
    </div>
  );
};
