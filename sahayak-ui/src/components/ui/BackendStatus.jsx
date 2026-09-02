/**
 * BackendStatus.jsx
 * Polls /health every few seconds. When the backend is down (Render cold start),
 * shows a friendly "waking up" banner instead of exposing the raw URL.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { getBackendUrl } from '../../api/client';

const POLL_INTERVAL_DOWN = 6000;   // poll every 6 s while down
const POLL_INTERVAL_UP   = 60000;  // re-check every 60 s while up

export const BackendStatus = () => {
  const [status, setStatus] = useState('checking'); // 'checking' | 'up' | 'waking' | 'down'
  const [dots, setDots] = useState('');
  const timerRef = useRef(null);
  const dotsRef = useRef(null);

  const check = useCallback(async () => {
    try {
      const url = `${getBackendUrl()}/health`;
      const resp = await fetch(url, { method: 'GET', signal: AbortSignal.timeout(8000) });
      if (resp.ok) {
        setStatus('up');
        timerRef.current = setTimeout(check, POLL_INTERVAL_UP);
      } else {
        setStatus('down');
        timerRef.current = setTimeout(check, POLL_INTERVAL_DOWN);
      }
    } catch {
      // Network error — likely Render cold start
      setStatus((prev) => (prev === 'checking' ? 'waking' : 'waking'));
      timerRef.current = setTimeout(check, POLL_INTERVAL_DOWN);
    }
  }, []);

  useEffect(() => {
    check();
    return () => {
      clearTimeout(timerRef.current);
      clearInterval(dotsRef.current);
    };
  }, [check]);

  // Animated dots while waking
  useEffect(() => {
    if (status === 'waking' || status === 'checking') {
      dotsRef.current = setInterval(() => {
        setDots((d) => (d.length >= 3 ? '' : d + '.'));
      }, 500);
    } else {
      clearInterval(dotsRef.current);
      setDots('');
    }
    return () => clearInterval(dotsRef.current);
  }, [status]);

  if (status === 'up') return null;

  const bannerConfig = {
    checking: {
      bg: 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-800/60',
      icon: '⏳',
      text: `Connecting to AI backend${dots}`,
      sub: 'Please wait a moment while we establish connection.',
    },
    waking: {
      bg: 'bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800/60',
      icon: '🔆',
      text: `Backend is starting up${dots}`,
      sub: 'The server is waking from sleep. This usually takes 30–60 seconds on the free tier.',
    },
    down: {
      bg: 'bg-red-50 dark:bg-red-950/40 border-red-200 dark:border-red-800/60',
      icon: '⚠️',
      text: 'Backend is temporarily unavailable',
      sub: 'We\'re unable to reach the AI server. Retrying automatically.',
    },
  };

  const cfg = bannerConfig[status] || bannerConfig.checking;

  return (
    <div className={`flex items-start gap-3 px-4 py-3 rounded-xl border text-sm ${cfg.bg} animate-fade-in`}>
      <span className="text-lg leading-none mt-0.5">{cfg.icon}</span>
      <div>
        <p className="font-semibold text-slate-800 dark:text-slate-100">{cfg.text}</p>
        <p className="text-slate-500 dark:text-slate-400 text-xs mt-0.5">{cfg.sub}</p>
      </div>
      {(status === 'waking' || status === 'checking') && (
        <div className="ml-auto flex items-center gap-1 shrink-0">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-amber-400 dark:bg-amber-500 animate-bounce"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      )}
    </div>
  );
};
