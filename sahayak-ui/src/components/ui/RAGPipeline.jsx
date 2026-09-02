/**
 * RAGPipeline.jsx
 * Animated step-by-step visualization of the RAG pipeline shown while a query runs.
 */
import React, { useEffect, useState } from 'react';

const STEPS = [
  { id: 'embed',    icon: '🔢', label: 'Embedding Query',     desc: 'Converting your question into a vector' },
  { id: 'search',   icon: '🔍', label: 'Vector Search',       desc: 'Finding similar chunks in Qdrant' },
  { id: 'context',  icon: '📄', label: 'Fetching Context',    desc: 'Retrieving relevant document passages' },
  { id: 'llm',      icon: '🧠', label: 'LLM Generation',      desc: 'Groq / OpenAI synthesizing your answer' },
  { id: 'done',     icon: '✅', label: 'Answer Ready',         desc: 'Response generated successfully' },
];

export const RAGPipeline = ({ active = true, done = false }) => {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (!active) { setCurrentStep(0); return; }
    if (done) { setCurrentStep(STEPS.length - 1); return; }

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        // Cycle through steps 0-3 while active (step 4 = done, set explicitly)
        const next = prev + 1;
        return next >= STEPS.length - 1 ? 0 : next;
      });
    }, 900);

    return () => clearInterval(interval);
  }, [active, done]);

  if (!active && !done) return null;

  return (
    <div className="mt-2 mb-1 px-3 py-2.5 rounded-xl bg-indigo-50/70 dark:bg-indigo-950/30 border border-indigo-200/60 dark:border-indigo-800/40 animate-fade-in">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-indigo-500 dark:text-indigo-400 mb-2">
        RAG Pipeline
      </p>
      <div className="flex items-center gap-1 overflow-x-auto pb-1">
        {STEPS.map((step, idx) => {
          const isActive = idx === currentStep && active && !done;
          const isPast   = done ? true : idx < currentStep;
          const isFuture = !done && idx > currentStep;

          return (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center min-w-[72px]">
                {/* Circle */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-base transition-all duration-500
                    ${isActive  ? 'ring-2 ring-indigo-500 ring-offset-2 dark:ring-offset-slate-900 scale-110 bg-indigo-100 dark:bg-indigo-900/60 shadow-md' : ''}
                    ${isPast    ? 'bg-emerald-100 dark:bg-emerald-900/40 opacity-80' : ''}
                    ${isFuture  ? 'bg-slate-100 dark:bg-slate-800 opacity-40' : ''}
                  `}
                >
                  {step.icon}
                </div>
                {/* Label */}
                <p className={`text-[9px] font-medium text-center mt-1 leading-tight transition-colors duration-300
                  ${isActive ? 'text-indigo-600 dark:text-indigo-400' : ''}
                  ${isPast   ? 'text-emerald-600 dark:text-emerald-400' : ''}
                  ${isFuture ? 'text-slate-400 dark:text-slate-600' : ''}
                `}>
                  {step.label}
                </p>
              </div>
              {/* Connector line */}
              {idx < STEPS.length - 1 && (
                <div className={`flex-1 h-0.5 rounded-full transition-all duration-500 min-w-[12px]
                  ${isPast && !isActive ? 'bg-emerald-400 dark:bg-emerald-600' : 'bg-slate-200 dark:bg-slate-700'}
                `} />
              )}
            </React.Fragment>
          );
        })}
      </div>
      {/* Current step description */}
      {active && !done && (
        <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1.5 italic">
          {STEPS[currentStep]?.desc}
        </p>
      )}
    </div>
  );
};
