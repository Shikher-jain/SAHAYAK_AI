import React from 'react';

export const PageHeader = ({
  title,
  subtitle,
  badge = null,
  action = null,
  className = '',
}) => {
  return (
    <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8 text-left ${className}`}>
      <div>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            {title}
          </h1>
          {badge}
        </div>
        {subtitle && (
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1.5 max-w-2xl leading-relaxed">
            {subtitle}
          </p>
        )}
      </div>
      {action && <div className="shrink-0 flex items-center gap-2">{action}</div>}
    </div>
  );
};
