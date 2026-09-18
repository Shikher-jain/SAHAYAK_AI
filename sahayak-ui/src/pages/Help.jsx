import React, { useState, useEffect } from 'react';
import { HelpCircle, Send, MessageSquare, ChevronDown, ChevronUp } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { PageHeader } from '../components/ui/PageHeader';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { callBackend } from '../api/client';
import { useAppContext } from '../context/AppContext';

export const Help = () => {
  const { showError } = useAppContext();
  const [faqs, setFaqs] = useState([]);
  const [loadingFaqs, setLoadingFaqs] = useState(true);
  const [expandedFaq, setExpandedFaq] = useState(null);

  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [asking, setAsking] = useState(false);

  useEffect(() => {
    fetchFaqs();
  }, []);

  const fetchFaqs = async () => {
    setLoadingFaqs(true);
    const { ok, data, error } = await callBackend('get', '/help/faq');
    if (ok && data) {
      // The Streamlit app expects a list of FAQs
      // Assuming data is an array or { faqs: [...] }
      setFaqs(Array.isArray(data) ? data : data.faqs || []);
    } else {
      showError(error || 'Failed to fetch FAQs');
    }
    setLoadingFaqs(false);
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setAsking(true);
    setAnswer('');
    
    const { ok, data, error } = await callBackend('post', '/help/ask', { question });
    if (ok && data) {
      setAnswer(data.answer || data.response || 'No answer provided.');
    } else {
      showError(error || 'Failed to get help');
    }
    setAsking(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <PageHeader 
        title="Help Center" 
        subtitle="Frequently Asked Questions and AI Support Assistant"
        icon={HelpCircle} 
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Column: FAQs */}
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">Frequently Asked Questions</h3>
          {loadingFaqs ? (
            <div className="animate-pulse space-y-3">
              {[1,2,3,4].map(i => <div key={i} className="h-16 bg-slate-200 dark:bg-slate-800 rounded-xl" />)}
            </div>
          ) : faqs.length > 0 ? (
            <div className="space-y-3">
              {faqs.map((faq, idx) => (
                <Card key={idx} className="p-0 overflow-hidden">
                  <button 
                    onClick={() => setExpandedFaq(expandedFaq === idx ? null : idx)}
                    className="w-full text-left p-4 flex justify-between items-center hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                  >
                    <span className="font-semibold text-sm text-slate-800 dark:text-slate-200">
                      {faq.question}
                    </span>
                    {expandedFaq === idx ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                  {expandedFaq === idx && (
                    <div className="p-4 pt-0 text-sm text-slate-600 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800">
                      {faq.answer}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">No FAQs available right now.</p>
          )}
        </div>

        {/* Right Column: AI Help Bot */}
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">Ask a Question</h3>
          <Card className="p-6">
            <form onSubmit={handleAsk} className="space-y-4">
              <Input
                label="How can we help?"
                placeholder="E.g., How do I upload a PDF?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                icon={MessageSquare}
              />
              <Button type="submit" loading={asking} icon={Send} className="w-full">
                {asking ? 'Asking...' : 'Ask AI Support'}
              </Button>
            </form>

            {answer && (
              <div className="mt-6 p-4 rounded-xl bg-indigo-50/50 dark:bg-indigo-900/20 border border-indigo-100 dark:border-indigo-800/50">
                <h4 className="font-bold text-sm text-indigo-800 dark:text-indigo-300 mb-2">Answer</h4>
                <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                  {answer}
                </p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};
