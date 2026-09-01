import React, { forwardRef } from 'react';

export const Input = forwardRef(({
  label,
  error,
  helperText,
  icon: Icon = null,
  rightElement = null,
  required = false,
  className = '',
  id,
  type = 'text',
  ...props
}, ref) => {
  const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

  return (
    <div className="flex flex-col gap-1.5 w-full text-left">
      {label && (
        <label htmlFor={inputId} className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
          <span>
            {label}
            {required && <span className="text-rose-500 ml-1" title="Required">*</span>}
          </span>
        </label>
      )}
      
      <div className="relative flex items-center">
        {Icon && (
          <div className="absolute left-3.5 text-slate-400 pointer-events-none flex items-center justify-center">
            <Icon size={18} />
          </div>
        )}
        
        <input
          ref={ref}
          id={inputId}
          type={type}
          required={required}
          className={`
            w-full px-3.5 py-2.5 rounded-xl text-sm transition-all duration-150
            bg-white dark:bg-slate-900 
            border ${error ? 'border-rose-400 dark:border-rose-500 focus:ring-rose-500/20' : 'border-slate-200 dark:border-slate-800 focus:border-indigo-500 dark:focus:border-indigo-500 focus:ring-indigo-500/20'}
            focus:outline-none focus:ring-4
            text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500
            disabled:opacity-60 disabled:cursor-not-allowed disabled:bg-slate-50 dark:disabled:bg-slate-950
            ${Icon ? 'pl-10' : ''}
            ${rightElement ? 'pr-11' : ''}
            ${className}
          `}
          {...props}
        />

        {rightElement && (
          <div className="absolute right-3 flex items-center">
            {rightElement}
          </div>
        )}
      </div>

      {error ? (
        <p className="text-xs text-rose-500 dark:text-rose-400 font-medium">{error}</p>
      ) : helperText ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">{helperText}</p>
      ) : null}
    </div>
  );
});

Input.displayName = 'Input';
