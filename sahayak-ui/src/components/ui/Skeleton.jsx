import React from 'react';

export const Skeleton = ({ className = '', variant = 'text', count = 1 }) => {
  const base = 'animate-pulse bg-slate-200 dark:bg-slate-800 rounded-lg';

  const variants = {
    text: 'h-4 w-full',
    title: 'h-7 w-1/3',
    circle: 'rounded-full w-10 h-10',
    card: 'h-32 w-full rounded-2xl',
    button: 'h-10 w-24 rounded-xl',
  };

  if (count > 1) {
    return (
      <div className="space-y-2.5 w-full">
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className={`${base} ${variants[variant] || ''} ${className}`} />
        ))}
      </div>
    );
  }

  return <div className={`${base} ${variants[variant] || ''} ${className}`} />;
};

export const CardSkeleton = () => (
  <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-6 space-y-4">
    <div className="flex items-center gap-4">
      <Skeleton variant="circle" />
      <div className="space-y-2 flex-1">
        <Skeleton variant="title" className="w-1/2" />
        <Skeleton variant="text" className="w-3/4" />
      </div>
    </div>
    <Skeleton variant="text" count={3} />
  </div>
);
