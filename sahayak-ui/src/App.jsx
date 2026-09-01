import React from 'react';
import { AppProvider, useAppContext } from './context/AppContext';
import { MainLayout } from './layouts/MainLayout';
import { AuthPage } from './pages/AuthPage';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { SearchChat } from './pages/SearchChat';
import { Counselor } from './pages/Counselor';
import { Quiz } from './pages/Quiz';
import { Roadmaps } from './pages/Roadmaps';
import { Books } from './pages/Books';
import { Progress } from './pages/Progress';
import { KnowledgeGraph } from './pages/KnowledgeGraph';
import { Stories } from './pages/Stories';
import { Pricing } from './pages/Pricing';
import { LearnHub } from './pages/LearnHub';
import { SettingsPage } from './pages/SettingsPage';

function AppContent() {
  const { authToken, currentPage } = useAppContext();

  // If user is not authenticated, show the modern Auth experience
  if (!authToken) {
    return <AuthPage />;
  }

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'upload':
        return <Upload />;
      case 'search':
        return <SearchChat />;
      case 'counselor':
        return <Counselor />;
      case 'quiz':
        return <Quiz />;
      case 'roadmaps':
        return <Roadmaps />;
      case 'books':
        return <Books />;
      case 'progress':
        return <Progress />;
      case 'knowledge':
        return <KnowledgeGraph />;
      case 'stories':
        return <Stories />;
      case 'pricing':
        return <Pricing />;
      case 'learn':
        return <LearnHub />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <MainLayout>
      {renderCurrentPage()}
    </MainLayout>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}