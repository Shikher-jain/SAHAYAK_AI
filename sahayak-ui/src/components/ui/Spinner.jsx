import React from 'react';
import { Loader2 } from 'lucide-react';

export const Spinner = ({ size = 'md', className = '', label = null }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12',
  };

  return (
    <div className={`inline-flex items-center justify-center gap-3 ${className}`}>
      <Loader2 className={`${sizeClasses[size] || sizeClasses.md} animate-spin text-indigo-600 dark:text-indigo-400`} />
      {label && <span className="text-sm font-medium text-slate-500 dark:text-slate-400">{label}</span>}
    </div>
  );
};
