import React from 'react';
import { Button } from './Button';

export const EmptyState = ({
  icon: Icon,
  title = 'No items found',
  description = 'There are no items to display at this time.',
  actionLabel = null,
  onAction = null,
  className = '',
}) => {
  return (
    <div className={`flex flex-col items-center justify-center p-12 text-center rounded-2xl border-2 border-dashed border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 ${className}`}>
      {Icon && (
        <div className="w-14 h-14 rounded-2xl bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-4 shadow-sm">
          <Icon size={28} />
        </div>
      )}
      <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-1">
        {title}
      </h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mb-6 leading-relaxed">
        {description}
      </p>
      {actionLabel && onAction && (
        <Button size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
};
