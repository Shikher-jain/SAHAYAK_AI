import React, { forwardRef } from 'react';
import { ChevronDown } from 'lucide-react';

export const Select = forwardRef(({
  label,
  error,
  helperText,
  options = [],
  required = false,
  className = '',
  id,
  children,
  ...props
}, ref) => {
  const selectId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

  return (
    <div className="flex flex-col gap-1.5 w-full text-left">
      {label && (
        <label htmlFor={selectId} className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          {label}
          {required && <span className="text-rose-500 ml-1">*</span>}
        </label>
      )}

      <div className="relative flex items-center">
        <select
          ref={ref}
          id={selectId}
          required={required}
          className={`
            w-full appearance-none px-3.5 py-2.5 rounded-xl text-sm transition-all duration-150
            bg-white dark:bg-slate-900 
            border ${error ? 'border-rose-400 dark:border-rose-500 focus:ring-rose-500/20' : 'border-slate-200 dark:border-slate-800 focus:border-indigo-500 dark:focus:border-indigo-500 focus:ring-indigo-500/20'}
            focus:outline-none focus:ring-4
            text-slate-900 dark:text-slate-100
            disabled:opacity-60 disabled:cursor-not-allowed
            pr-10
            ${className}
          `}
          {...props}
        >
          {children ? (
            children
          ) : (
            options.map((opt) => {
              const value = typeof opt === 'object' ? opt.value : opt;
              const labelText = typeof opt === 'object' ? opt.label : opt;
              return (
                <option key={value} value={value} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                  {labelText}
                </option>
              );
            })
          )}
        </select>
        <div className="absolute right-3 pointer-events-none text-slate-400">
          <ChevronDown size={18} />
        </div>
      </div>

      {error ? (
        <p className="text-xs text-rose-500 dark:text-rose-400 font-medium">{error}</p>
      ) : helperText ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">{helperText}</p>
      ) : null}
    </div>
  );
});

Select.displayName = 'Select';
