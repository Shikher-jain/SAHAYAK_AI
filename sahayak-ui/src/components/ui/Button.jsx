import React from 'react';
import { Loader2 } from 'lucide-react';

export const Button = ({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  className = '',
  disabled = false,
  loading = false,
  type = 'button',
  icon: Icon = null,
  iconPosition = 'left',
  title,
  ...props
}) => {
  const baseStyle = 'inline-flex items-center justify-center font-medium rounded-xl transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none disabled:active:scale-100 select-none';

  const sizes = {
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-2.5 text-base gap-2.5',
    icon: 'p-2 rounded-xl',
  };

  const variants = {
    primary: 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm hover:shadow focus-visible:ring-indigo-500 dark:bg-indigo-600 dark:hover:bg-indigo-500 dark:focus-visible:ring-offset-slate-900',
    secondary: 'bg-slate-100 hover:bg-slate-200 text-slate-800 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-200 focus-visible:ring-slate-400 dark:focus-visible:ring-offset-slate-900',
    outline: 'border border-slate-200 hover:border-slate-300 dark:border-slate-700 dark:hover:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/60 focus-visible:ring-slate-400 dark:focus-visible:ring-offset-slate-900',
    ghost: 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800/60 focus-visible:ring-slate-400',
    danger: 'bg-rose-50 hover:bg-rose-100 text-rose-600 border border-rose-200 dark:bg-rose-950/40 dark:hover:bg-rose-900/50 dark:text-rose-400 dark:border-rose-900/50 focus-visible:ring-rose-500',
    success: 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm focus-visible:ring-emerald-500',
    subtle: 'bg-indigo-50 hover:bg-indigo-100 text-indigo-700 dark:bg-indigo-950/40 dark:hover:bg-indigo-900/50 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/40',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      title={title}
      className={`${baseStyle} ${sizes[size] || sizes.md} ${variants[variant] || variants.primary} ${className}`}
      {...props}
    >
      {loading ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin shrink-0" />
          {children && <span>{children}</span>}
        </>
      ) : (
        <>
          {Icon && iconPosition === 'left' && <Icon className="w-4 h-4 shrink-0" />}
          {children}
          {Icon && iconPosition === 'right' && <Icon className="w-4 h-4 shrink-0" />}
        </>
      )}
    </button>
  );
};
