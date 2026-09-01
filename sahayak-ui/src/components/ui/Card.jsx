import React from 'react';

export const Card = ({
  children,
  className = '',
  hoverable = false,
  onClick,
  ...props
}) => {
  return (
    <div
      onClick={onClick}
      className={`
        bg-white dark:bg-slate-900 
        border border-slate-200/80 dark:border-slate-800/80 
        rounded-2xl p-6 shadow-sm
        transition-all duration-200
        ${hoverable ? 'hover:shadow-card-hover hover:border-slate-300 dark:hover:border-slate-700 cursor-pointer' : ''}
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
};

export const CardHeader = ({ title, subtitle, action, className = '' }) => (
  <div className={`flex items-start justify-between gap-4 mb-5 ${className}`}>
    <div>
      <h3 className="font-semibold text-base text-slate-900 dark:text-slate-100">{title}</h3>
      {subtitle && <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{subtitle}</p>}
    </div>
    {action && <div className="shrink-0">{action}</div>}
  </div>
);
