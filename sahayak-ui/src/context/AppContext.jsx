import React, { createContext, useState, useEffect, useCallback } from 'react';


const UI_LABELS = {
  en: {
    dashboard: "Dashboard",
    learn: "Study Hub",
    upload: "Upload Documents",
    search: "Search & Chat",
    counselor: "AI Counselor",
    quiz: "Quiz Engine",
    knowledge: "Knowledge Graph",
    roadmaps: "Roadmaps",
    books: "Books & Resources",
    progress: "My Progress",
    stories: "Community Stories",
    pricing: "Pricing & Plans",
    settings: "Settings",
    profile: "Profile",
    help: "Help & FAQ",
    contact: "Contact Us",
    about: "About Sahayak",
    sync: "Sync Data"
  },
  hi: {
    dashboard: "डैशबोर्ड",
    learn: "अध्ययन केंद्र",
    upload: "दस्तावेज़ अपलोड",
    search: "खोजें और चैट",
    counselor: "AI परामर्शदाता",
    quiz: "प्रश्नोत्तरी (क्विज़)",
    knowledge: "ज्ञान ग्राफ़",
    roadmaps: "अध्ययन रोडमैप",
    books: "पुस्तकें और संसाधन",
    progress: "मेरी प्रगति",
    stories: "समुदाय की कहानियाँ",
    pricing: "योजनाएं व मूल्य",
    settings: "सेटिंग्स",
    profile: "प्रोफ़ाइल",
    help: "सहायता",
    contact: "संपर्क करें",
    about: "सहायक के बारे में",
    sync: "डेटा सिंक"
  },
  es: {
    dashboard: "Panel",
    learn: "Centro de Estudio",
    upload: "Subir Documentos",
    search: "Buscar y Chat",
    counselor: "Consejero IA",
    quiz: "Cuestionarios",
    knowledge: "Grafo de Conocimiento",
    roadmaps: "Rutas de Aprendizaje",
    books: "Libros y Recursos",
    progress: "Mi Progreso",
    stories: "Historias",
    pricing: "Precios",
    settings: "Configuración",
    profile: "Perfil",
    help: "Ayuda",
    contact: "Contacto",
    about: "Acerca de",
    sync: "Sincronizar"
  },
  fr: {
    dashboard: "Tableau de bord",
    learn: "Centre d'étude",
    upload: "Téléverser",
    search: "Recherche & Chat",
    counselor: "Conseiller IA",
    quiz: "Quiz",
    knowledge: "Graphe de Connaissances",
    roadmaps: "Parcours",
    books: "Livres & Ressources",
    progress: "Ma Progression",
    stories: "Histoires",
    pricing: "Tarifs",
    settings: "Paramètres",
    profile: "Profil",
    help: "Aide",
    contact: "Contact",
    about: "À propos",
    sync: "Synchroniser"
  },
  de: {
    dashboard: "Übersicht",
    learn: "Lernzentrum",
    upload: "Dokumente hochladen",
    search: "Suche & Chat",
    counselor: "KI-Berater",
    quiz: "Quiz",
    knowledge: "Wissensgraph",
    roadmaps: "Lernpfade",
    books: "Bücher & Ressourcen",
    progress: "Mein Fortschritt",
    stories: "Erfahrungsberichte",
    pricing: "Preise",
    settings: "Einstellungen",
    profile: "Profil",
    help: "Hilfe",
    contact: "Kontakt",
    about: "Über uns",
    sync: "Synchronisieren"
  }
};

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  // Theme state
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('sahayak_theme');
    if (saved) return saved;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  // Language state
  const [language, setLanguage] = useState(() => localStorage.getItem('sahayak_language') || 'en');
  
  // Navigation state
  const [currentPage, setCurrentPage] = useState('dashboard');
  
  // RAG / AI Session
  const [ragSessionId, setRagSessionId] = useState(() => {
    const existing = sessionStorage.getItem('sahayak_rag_session');
    if (existing) return existing;
    const newId = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem('sahayak_rag_session', newId);
    return newId;
  });

  // Learning / Persona state
  const [learningMode, setLearningMode] = useState(() => localStorage.getItem('sahayak_learning_mode') || 'student');
  const [userMode, setUserMode] = useState(() => localStorage.getItem('sahayak_user_mode') || 'general');

  // Auth state
  const [authToken, setAuthToken] = useState(() => localStorage.getItem('auth_token'));
  const [authUser, setAuthUser] = useState(() => localStorage.getItem('auth_user') || 'Learner');
  const [authRole, setAuthRole] = useState(() => localStorage.getItem('auth_role') || 'student');

  // Toast notification state
  const [toasts, setToasts] = useState([]);

  // Theme effect
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('sahayak_theme', theme);
  }, [theme]);

  // Language effect
  useEffect(() => {
    localStorage.setItem('sahayak_language', language);
  }, [language]);

  // Persona effects
  useEffect(() => {
    localStorage.setItem('sahayak_learning_mode', learningMode);
  }, [learningMode]);

  useEffect(() => {
    localStorage.setItem('sahayak_user_mode', userMode);
  }, [userMode]);

  // Toast helpers
  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = Date.now() + Math.random().toString(36).substr(2, 5);
    setToasts((prev) => [...prev, { id, message, type, duration }]);
    return id;
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showSuccess = useCallback((msg) => addToast(msg, 'success'), [addToast]);
  const showError = useCallback((msg) => addToast(msg, 'error', 5000), [addToast]);
  const showInfo = useCallback((msg) => addToast(msg, 'info'), [addToast]);
  const showWarning = useCallback((msg) => addToast(msg, 'warning', 4500), [addToast]);

  // Auth actions
  const login = (token, username, role = 'student') => {
    localStorage.setItem('auth_token', token);
    localStorage.setItem('auth_user', username);
    localStorage.setItem('auth_role', role);
    setAuthToken(token);
    setAuthUser(username);
    setAuthRole(role);
    setCurrentPage('dashboard');
    showSuccess(`Welcome back, ${username}!`);
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    localStorage.removeItem('auth_role');
    setAuthToken(null);
    setAuthUser('Learner');
    setAuthRole('student');
    showInfo('You have been signed out.');
  };

  const newRagSession = () => {
    const newId = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem('sahayak_rag_session', newId);
    setRagSessionId(newId);
    return newId;
  };

  // Translation helper
  const t = (key) => {
    const langDict = UI_LABELS[language] || UI_LABELS.en;
    return langDict[key] || UI_LABELS.en[key] || key;
  };

  const value = {
    theme,
    setTheme,
    toggleTheme: () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark')),
    language,
    setLanguage,
    currentPage,
    setCurrentPage,
    ragSessionId,
    newRagSession,
    learningMode,
    setLearningMode,
    userMode,
    setUserMode,
    authToken,
    authUser,
    authRole,
    login,
    logout,
    toasts,
    addToast,
    removeToast,
    showSuccess,
    showError,
    showInfo,
    showWarning,
    t
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export { AppContext };
export { useAppContext } from './useAppContext';

