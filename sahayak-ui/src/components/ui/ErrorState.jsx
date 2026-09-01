import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './Button';

export const ErrorState = ({
  title = 'Something went wrong',
  error = null,
  onRetry = null,
  className = '',
}) => {
  return (
    <div className={`p-6 rounded-2xl bg-rose-50/70 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/40 text-left ${className}`}>
      <div className="flex items-start gap-4">
        <div className="p-2.5 rounded-xl bg-rose-100 dark:bg-rose-900/40 text-rose-600 dark:text-rose-400 shrink-0">
          <AlertCircle size={22} />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-rose-900 dark:text-rose-200 text-sm">
            {title}
          </h3>
          {error && (
            <p className="text-xs text-rose-700 dark:text-rose-300/80 mt-1 font-mono break-words">
              {typeof error === 'object' ? JSON.stringify(error) : error}
            </p>
          )}
          {onRetry && (
            <div className="mt-4">
              <Button
                size="sm"
                variant="danger"
                icon={RefreshCw}
                onClick={onRetry}
              >
                Try Again
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
