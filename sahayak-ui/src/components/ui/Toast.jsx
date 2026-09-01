import React, { useEffect } from 'react';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';
import { useAppContext } from '../../context/AppContext';

const ToastItem = ({ toast }) => {
  const { removeToast } = useAppContext();

  useEffect(() => {
    if (toast.duration) {
      const timer = setTimeout(() => {
        removeToast(toast.id);
      }, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast, removeToast]);

  const icons = {
    success: CheckCircle2,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
  };

  const colors = {
    success: 'bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-800',
    error: 'bg-rose-50 text-rose-800 border-rose-200 dark:bg-rose-950 dark:text-rose-200 dark:border-rose-800',
    warning: 'bg-amber-50 text-amber-800 border-amber-200 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800',
    info: 'bg-indigo-50 text-indigo-800 border-indigo-200 dark:bg-indigo-950 dark:text-indigo-200 dark:border-indigo-800',
  };

  const IconComponent = icons[toast.type] || Info;

  return (
    <div
      className={`
        flex items-start gap-3 p-4 rounded-xl border shadow-lg backdrop-blur-md 
        animate-slide-up transition-all duration-200 min-w-[300px] max-w-md
        ${colors[toast.type] || colors.info}
      `}
      role="alert"
    >
      <IconComponent size={20} className="shrink-0 mt-0.5" />
      <div className="flex-1 text-xs font-medium leading-relaxed">
        {toast.message}
      </div>
      <button
        onClick={() => removeToast(toast.id)}
        className="opacity-70 hover:opacity-100 p-0.5 rounded transition-opacity"
        aria-label="Close notification"
      >
        <X size={16} />
      </button>
    </div>
  );
};

export const ToastContainer = () => {
  const { toasts } = useAppContext();

  if (!toasts || toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-auto">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  );
};
