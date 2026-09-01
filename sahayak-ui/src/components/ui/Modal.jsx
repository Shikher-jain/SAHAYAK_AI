import React, { useEffect } from 'react';
import { X } from 'lucide-react';


export const Modal = ({
  isOpen,
  onClose,
  title,
  subtitle,
  children,
  maxWidth = 'max-w-lg',
  actions = null,
}) => {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity animate-fade-in"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal Box */}
      <div
        className={`
          relative w-full ${maxWidth} 
          bg-white dark:bg-slate-900 
          border border-slate-200 dark:border-slate-800 
          rounded-2xl shadow-2xl overflow-hidden
          animate-slide-up z-10 my-8
        `}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'modal-title' : undefined}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-slate-100 dark:border-slate-800">
          <div>
            {title && (
              <h2 id="modal-title" className="text-lg font-bold text-slate-900 dark:text-slate-100">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                {subtitle}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            aria-label="Close modal"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 max-h-[calc(85vh-10rem)] overflow-y-auto">
          {children}
        </div>

        {/* Footer actions */}
        {actions && (
          <div className="flex items-center justify-end gap-3 p-4 bg-slate-50 dark:bg-slate-950/50 border-t border-slate-100 dark:border-slate-800">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
};
