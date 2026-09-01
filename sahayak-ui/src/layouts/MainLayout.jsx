import React, { useState } from 'react';
import { 
  Home, UploadCloud, Search, BookOpen, Settings, LogOut, Menu, X, Moon, Sun, 
  Bot, CheckSquare, Map, Book, Briefcase, BarChart, Heart, CreditCard, 
  HelpCircle, Share2, Sparkles, Globe, ChevronDown, Check, Shield
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { ToastContainer } from '../components/ui/Toast';
import { Modal } from '../components/ui/Modal';
import { Button } from '../components/ui/Button';

export const MainLayout = ({ children }) => {
  const { 
    t, 
    currentPage, 
    setCurrentPage, 
    authUser, 
    authRole, 
    logout, 
    theme, 
    toggleTheme, 
    language, 
    setLanguage,
    userMode,
    setUserMode
  } = useAppContext();

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [showLangMenu, setShowLangMenu] = useState(false);
  const [showPersonaMenu, setShowPersonaMenu] = useState(false);


  const navGroups = [
    {
      label: "Core Modules",
      items: [
        { id: 'dashboard', name: t('dashboard'), icon: Home, badge: null },
        { id: 'upload', name: t('upload'), icon: UploadCloud, badge: 'RAG' },
        { id: 'search', name: t('search'), icon: Search, badge: 'AI' },
      ]
    },
    {
      label: "Study & Practice",
      items: [
        { id: 'learn', name: t('learn'), icon: BookOpen },
        { id: 'quiz', name: t('quiz'), icon: CheckSquare, badge: 'Adaptive' },
        { id: 'knowledge', name: t('knowledge'), icon: Share2 },
      ]
    },
    {
      label: "Resources & Guidance",
      items: [
        { id: 'roadmaps', name: t('roadmaps'), icon: Map },
        { id: 'books', name: t('books'), icon: Book },
        { id: 'counselor', name: t('counselor'), icon: Briefcase, badge: 'Mentor' },
      ]
    },
    {
      label: "Community & Account",
      items: [
        { id: 'progress', name: t('progress'), icon: BarChart },
        { id: 'stories', name: t('stories'), icon: Heart },
        { id: 'pricing', name: t('pricing'), icon: CreditCard },
        { id: 'settings', name: t('settings'), icon: Settings },
      ]
    }
  ];

  const languages = [
    { code: 'en', label: 'English', native: 'English' },
    { code: 'hi', label: 'Hindi', native: 'हिन्दी' },
    { code: 'es', label: 'Spanish', native: 'Español' },
    { code: 'fr', label: 'French', native: 'Français' },
    { code: 'de', label: 'German', native: 'Deutsch' },
  ];

  const personas = [
    { id: 'student', label: 'Student Mode', desc: 'Detailed, step-by-step educational explanations' },
    { id: 'teacher', label: 'Teacher Mode', desc: 'Pedagogical insights & curriculum-oriented prompts' },
    { id: 'general', label: 'General / Professional', desc: 'Concise, direct, and actionable summaries' },
  ];

  const handleNavClick = (pageId) => {
    setCurrentPage(pageId);
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-[#0b0f19] text-slate-900 dark:text-slate-100 font-sans overflow-hidden">
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm lg:hidden transition-opacity" 
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside 
        className={`
          fixed lg:static inset-y-0 left-0 z-50 w-72 
          bg-white dark:bg-slate-900/95 
          border-r border-slate-200/80 dark:border-slate-800/80 
          transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'} 
          flex flex-col shadow-xl lg:shadow-none
        `}
      >
        {/* Brand Header */}
        <div className="flex items-center justify-between h-20 px-6 border-b border-slate-200/70 dark:border-slate-800/70 shrink-0">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => handleNavClick('dashboard')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
              <Bot size={22} />
            </div>
            <div>
              <div className="font-bold text-base tracking-tight text-slate-900 dark:text-white flex items-center gap-1.5">
                Sahayak AI
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 font-bold uppercase tracking-wider">
                  v2.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">Multimodal AI Learning</p>
            </div>
          </div>

          <button 
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 overflow-y-auto px-4 py-5 space-y-6">
          {navGroups.map((group, gIdx) => (
            <div key={gIdx}>
              <div className="text-[11px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2 px-3">
                {group.label}
              </div>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const isActive = currentPage === item.id;
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleNavClick(item.id)}
                      className={`
                        w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-semibold
                        transition-all duration-150 group text-left
                        ${isActive 
                          ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 shadow-sm border border-indigo-100/60 dark:border-indigo-800/40' 
                          : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100/80 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-200'}
                      `}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <Icon className={`w-4 h-4 shrink-0 transition-colors ${isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-300'}`} />
                        <span className="truncate">{item.name}</span>
                      </div>
                      {item.badge && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-md font-medium ${isActive ? 'bg-indigo-200/50 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-200' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'}`}>
                          {item.badge}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* User Card & Footer */}
        <div className="p-4 border-t border-slate-200/70 dark:border-slate-800/70 shrink-0">
          <div className="flex items-center justify-between p-2 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800/60">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-9 h-9 rounded-xl bg-indigo-600/10 dark:bg-indigo-400/10 text-indigo-600 dark:text-indigo-400 flex items-center justify-center font-bold text-sm shrink-0 border border-indigo-200/30 dark:border-indigo-700/30">
                {authUser ? authUser.charAt(0).toUpperCase() : 'U'}
              </div>
              <div className="truncate text-left">
                <p className="text-xs font-bold text-slate-900 dark:text-white truncate">{authUser}</p>
                <p className="text-[10px] text-slate-400 capitalize flex items-center gap-1">
                  <Shield size={10} className="text-indigo-500" />
                  {authRole} Plan
                </p>
              </div>
            </div>
            <button
              onClick={logout}
              title="Sign Out"
              className="p-1.5 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 rounded-lg hover:bg-white dark:hover:bg-slate-700 transition-colors"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Top Navbar */}
        <header className="flex items-center justify-between h-20 px-6 bg-white/70 dark:bg-slate-950/70 backdrop-blur-md border-b border-slate-200/70 dark:border-slate-800/70 z-10 shrink-0">
          <div className="flex items-center gap-3">
            <button 
              className="lg:hidden p-2 -ml-2 rounded-xl text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors" 
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
            >
              <Menu size={22} />
            </button>
            <div className="hidden sm:flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
              <span className="hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer" onClick={() => setCurrentPage('dashboard')}>Sahayak</span>
              <span>/</span>
              <span className="text-slate-900 dark:text-slate-100 capitalize">{currentPage.replace('-', ' ')}</span>
            </div>
          </div>

          {/* Header Controls */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Persona Switcher Dropdown */}
            <div className="relative">
              <button
                onClick={() => {
                  setShowPersonaMenu(!showPersonaMenu);
                  setShowLangMenu(false);
                }}
                className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 border border-indigo-200/50 dark:border-indigo-800/50 hover:bg-indigo-100 transition-colors"
              >
                <Sparkles size={14} className="text-indigo-500" />
                <span className="hidden md:inline capitalize">{userMode} Mode</span>
                <ChevronDown size={14} />
              </button>

              {showPersonaMenu && (
                <div className="absolute right-0 mt-2 w-64 p-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-xl z-50 text-left animate-slide-up">
                  <div className="px-3 py-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    AI Response Persona
                  </div>
                  {personas.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setUserMode(p.id);
                        setShowPersonaMenu(false);
                      }}
                      className={`w-full p-2.5 rounded-xl text-left transition-colors flex items-start justify-between ${userMode === p.id ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300' : 'hover:bg-slate-50 dark:hover:bg-slate-800'}`}
                    >
                      <div>
                        <div className="text-xs font-bold">{p.label}</div>
                        <div className="text-[11px] text-slate-400 leading-tight mt-0.5">{p.desc}</div>
                      </div>
                      {userMode === p.id && <Check size={14} className="text-indigo-600 shrink-0 mt-0.5" />}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Language Switcher */}
            <div className="relative">
              <button
                onClick={() => {
                  setShowLangMenu(!showLangMenu);
                  setShowPersonaMenu(false);
                }}
                className="p-2 rounded-xl text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors flex items-center gap-1.5"
                title="Change Language"
              >
                <Globe size={18} />
                <span className="text-xs font-semibold uppercase hidden sm:inline">{language}</span>
              </button>

              {showLangMenu && (
                <div className="absolute right-0 mt-2 w-44 p-1.5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-xl z-50 text-left animate-slide-up">
                  {languages.map((l) => (
                    <button
                      key={l.code}
                      onClick={() => {
                        setLanguage(l.code);
                        setShowLangMenu(false);
                      }}
                      className={`w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center justify-between ${language === l.code ? 'bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400' : 'hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'}`}
                    >
                      <span>{l.native}</span>
                      <span className="text-[10px] text-slate-400 uppercase">{l.code}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-xl text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
              aria-label="Toggle Theme"
            >
              {theme === 'dark' ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} className="text-slate-600" />}
            </button>

            {/* Help / Guide Trigger */}
            <button
              onClick={() => setShowHelpModal(true)}
              className="p-2 rounded-xl text-slate-500 hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Help & Quick Tips"
            >
              <HelpCircle size={18} />
            </button>
          </div>
        </header>

        {/* Scrollable Page Body */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8">
          {children}
        </div>

        {/* Floating Quick Ask / Help Button */}
        <button
          onClick={() => setCurrentPage('search')}
          className="fixed bottom-6 right-6 w-12 h-12 bg-indigo-600 hover:bg-indigo-700 text-white rounded-2xl shadow-lg hover:shadow-indigo-500/25 hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center z-30 group"
          title="Open AI Assistant"
        >
          <Bot size={24} className="group-hover:scale-110 transition-transform" />
        </button>

        {/* App-wide Toast stack */}
        <ToastContainer />

        {/* Help Modal */}
        <Modal
          isOpen={showHelpModal}
          onClose={() => setShowHelpModal(false)}
          title="Sahayak AI Quick Guide"
          subtitle="Explore the capabilities of your multimodal educational companion."
          actions={
            <Button onClick={() => setShowHelpModal(false)} size="sm">
              Got it, let's learn!
            </Button>
          }
        >
          <div className="space-y-4 text-xs text-slate-600 dark:text-slate-300 text-left">
            <div className="p-3.5 rounded-xl bg-indigo-50/70 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-900/40">
              <h4 className="font-bold text-indigo-900 dark:text-indigo-200 mb-1">1. Multimodal Document Ingestion</h4>
              <p>Upload PDFs, Word documents, text notes, or web URLs in the <strong>Upload</strong> section. They will be chunked and indexed into vector embeddings.</p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800/70">
              <h4 className="font-bold text-slate-900 dark:text-white mb-1">2. Grounded RAG Search & Chat</h4>
              <p>Ask questions in any language in <strong>Search & Chat</strong>. Answers will cite specific source files and provide follow-up learning suggestions.</p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800/70">
              <h4 className="font-bold text-slate-900 dark:text-white mb-1">3. Adaptive Quiz Engine</h4>
              <p>Generate interactive quizzes on any syllabus topic in the <strong>Quiz Engine</strong> to test your mastery and receive immediate answer explanations.</p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800/70">
              <h4 className="font-bold text-slate-900 dark:text-white mb-1">4. AI Career & Academic Counselor</h4>
              <p>Consult with domain-specialized AI mentors (STEM, Arts, Commerce, Medical, Law) for personalized guidance and curated career roadmaps.</p>
            </div>
          </div>
        </Modal>
      </main>
    </div>
  );
};
