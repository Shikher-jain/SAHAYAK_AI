import React, { useState } from 'react';
import { 
  Sun, Moon, Globe, Database, Key, LogOut, Check, RotateCcw, Sparkles 
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { getBackendUrl, setBackendUrl, DEFAULT_BACKEND_URL } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';

export const SettingsPage = () => {
  const { 
    theme, 
    setTheme, 
    language, 
    setLanguage, 
    userMode, 
    setUserMode, 
    authUser, 
    authRole, 
    logout, 
    showSuccess 
  } = useAppContext();

  const [customBackendUrl, setCustomBackendUrl] = useState(getBackendUrl());
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('api_key') || '');

  const handleSaveApiSettings = (e) => {
    e.preventDefault();
    setBackendUrl(customBackendUrl);
    if (apiKey.trim()) {
      localStorage.setItem('api_key', apiKey.trim());
    } else {
      localStorage.removeItem('api_key');
    }
    showSuccess('API and Backend settings saved successfully!');
  };

  const handleResetBackend = () => {
    setCustomBackendUrl(DEFAULT_BACKEND_URL);
    setBackendUrl(DEFAULT_BACKEND_URL);
    showSuccess('Backend URL reset to default (http://127.0.0.1:8000).');
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Settings & Preferences"
        subtitle="Customize interface appearance, AI persona modes, language, and backend network connectivity."
        badge={<Badge variant="neutral" size="md">Preferences</Badge>}
      />

      {/* Appearance & Interface */}
      <Card className="p-6 sm:p-7 space-y-6">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Sun size={18} className="text-amber-500" />
          Appearance & Theme
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setTheme('light')}
            className={`flex items-center justify-center gap-2.5 p-4 rounded-2xl border font-bold text-xs transition-all ${theme === 'light' ? 'bg-indigo-50 border-indigo-500 text-indigo-900 shadow-sm' : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300'}`}
          >
            <Sun size={18} className="text-amber-500" />
            <span>Light Mode</span>
            {theme === 'light' && <Check size={16} className="text-indigo-600 ml-auto" />}
          </button>

          <button
            type="button"
            onClick={() => setTheme('dark')}
            className={`flex items-center justify-center gap-2.5 p-4 rounded-2xl border font-bold text-xs transition-all ${theme === 'dark' ? 'bg-indigo-950/70 border-indigo-500 text-indigo-200 shadow-sm' : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300'}`}
          >
            <Moon size={18} className="text-indigo-400" />
            <span>Dark Mode</span>
            {theme === 'dark' && <Check size={16} className="text-indigo-400 ml-auto" />}
          </button>
        </div>
      </Card>

      {/* Language & Localization */}
      <Card className="p-6 sm:p-7 space-y-6">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Globe size={18} className="text-indigo-500" />
          Language & Localization
        </h3>

        <Select
          label="Interface Language"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          options={[
            { value: 'en', label: 'English (Default)' },
            { value: 'hi', label: 'हिन्दी (Hindi)' },
            { value: 'es', label: 'Español (Spanish)' },
            { value: 'fr', label: 'Français (French)' },
            { value: 'de', label: 'Deutsch (German)' }
          ]}
          helperText="Translates navigation labels and prompts across the Sahayak platform."
        />
      </Card>

      {/* AI Persona */}
      <Card className="p-6 sm:p-7 space-y-6">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Sparkles size={18} className="text-indigo-500" />
          AI Tutor Persona (User Mode)
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            { id: 'student', title: 'Student', desc: 'Step-by-step educational explanations with examples' },
            { id: 'teacher', title: 'Teacher', desc: 'Pedagogical syllabus insights and class strategies' },
            { id: 'general', title: 'General', desc: 'Concise, direct, and actionable answers' },
          ].map((mode) => (
            <button
              key={mode.id}
              type="button"
              onClick={() => setUserMode(mode.id)}
              className={`p-4 rounded-2xl border text-left transition-all ${userMode === mode.id ? 'bg-indigo-50/80 dark:bg-indigo-950/70 border-indigo-500 text-indigo-900 dark:text-indigo-200' : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300'}`}
            >
              <p className="font-bold text-xs capitalize flex items-center justify-between">
                <span>{mode.title} Mode</span>
                {userMode === mode.id && <Check size={14} className="text-indigo-600 dark:text-indigo-400" />}
              </p>
              <p className="text-[11px] text-slate-400 mt-1 leading-snug">{mode.desc}</p>
            </button>
          ))}
        </div>
      </Card>

      {/* Backend & API Configuration */}
      <Card className="p-6 sm:p-7 space-y-6">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Database size={18} className="text-emerald-500" />
          Backend API Connection
        </h3>

        <form onSubmit={handleSaveApiSettings} className="space-y-4">
          <Input
            label="FastAPI Backend Endpoint URL"
            value={customBackendUrl}
            onChange={(e) => setCustomBackendUrl(e.target.value)}
            helperText="Default is http://127.0.0.1:8000. Point to production host if deployed remotely."
            required
          />

          <Input
            label="API Key (Optional Header: X-API-Key)"
            type="password"
            placeholder="Enter SAHAYAK_API_KEY if enforced on backend..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            icon={Key}
          />

          <div className="pt-2 flex flex-wrap gap-3 items-center justify-between">
            <Button
              type="button"
              variant="outline"
              size="sm"
              icon={RotateCcw}
              onClick={handleResetBackend}
            >
              Reset to Default
            </Button>

            <Button type="submit" size="sm">
              Save Connection Settings
            </Button>
          </div>
        </form>
      </Card>

      {/* Account & Sign Out */}
      <Card className="p-6 sm:p-7 border-rose-100 dark:border-rose-950/50 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">
              Signed in as <span className="text-indigo-600 dark:text-indigo-400">{authUser}</span>
            </h3>
            <p className="text-xs text-slate-400">
              Role: <span className="capitalize font-semibold">{authRole}</span>
            </p>
          </div>
          <Button
            variant="danger"
            size="sm"
            icon={LogOut}
            onClick={logout}
          >
            Sign Out
          </Button>
        </div>
      </Card>
    </div>
  );
};
