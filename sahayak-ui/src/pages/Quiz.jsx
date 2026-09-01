import React, { useState } from 'react';
import { 
  CheckSquare, Sparkles, Award, CheckCircle2, XCircle, 
  RotateCcw
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { ErrorState } from '../components/ui/ErrorState';


export const Quiz = () => {
  const { showSuccess, showError } = useAppContext();
  const [topic, setTopic] = useState('');
  const [numQuestions, setNumQuestions] = useState(5);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [quiz, setQuiz] = useState(null);
  const [userAnswers, setUserAnswers] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const sampleTopics = [
    "Photosynthesis & Plant Biology",
    "Python OOP & Data Structures",
    "Indian Constitution & Preamble",
    "Calculus: Derivatives & Integrals",
    "Principles of Macroeconomics",
    "Machine Learning Foundations"
  ];

  const handleGenerate = async (selectedTopic) => {
    const activeTopic = (selectedTopic || topic).trim();
    if (!activeTopic) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const { ok, data, error: err } = await callBackend('post', '/quiz/generate', {
      topic: activeTopic,
      num_questions: numQuestions
    });

    if (ok && data && Array.isArray(data.questions) && data.questions.length > 0) {
      setQuiz(data);
      setUserAnswers(Array(data.questions.length).fill(-1));
      showSuccess(`Generated ${data.questions.length} questions on ${activeTopic}!`);
    } else {
      setError(err || 'Failed to generate quiz. Make sure the backend LLM service is available.');
      showError(err || 'Failed to generate quiz');
    }
    setLoading(false);
  };

  const handleSelectOption = (questionIndex, optionIndex) => {
    const updated = [...userAnswers];
    updated[questionIndex] = optionIndex;
    setUserAnswers(updated);
  };

  const handleSubmit = async () => {
    if (!quiz || userAnswers.includes(-1)) {
      showError('Please answer all questions before submitting.');
      return;
    }

    setSubmitting(true);
    setError(null);

    const { ok, data } = await callBackend('post', '/quiz/answer', {
      topic: quiz.topic,
      questions: quiz.questions,
      answers: userAnswers
    });


    if (ok && data) {
      setResult(data);
      showSuccess(`Quiz evaluated! Your score: ${data.correct || 0}/${data.total || quiz.questions.length}`);
    } else {
      // If endpoint fails, calculate client-side fallback score
      const evaluated = quiz.questions.map((q, idx) => {
        const correctIndex = q.correct_answer ?? q.answer_index ?? 0;
        const isCorrect = userAnswers[idx] === correctIndex;
        return {
          question_index: idx,
          user_answer: userAnswers[idx],
          correct: isCorrect,
          explanation: q.explanation || `The correct option is: ${q.options[correctIndex] || 'Option A'}`
        };
      });
      const correctCount = evaluated.filter(e => e.correct).length;
      setResult({
        topic: quiz.topic,
        total: quiz.questions.length,
        correct: correctCount,
        score: correctCount / quiz.questions.length,
        results: evaluated
      });
    }
    setSubmitting(false);
  };

  const answeredCount = userAnswers.filter(a => a !== -1).length;
  const isComplete = quiz && answeredCount === quiz.questions.length;

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Adaptive Quiz Engine"
        subtitle="Test your comprehension with dynamically synthesized, curriculum-aligned multiple choice tests."
        badge={<Badge variant="primary" size="md">Adaptive AI</Badge>}
      />

      {error && (
        <ErrorState
          title="Quiz Engine Notice"
          error={error}
          onRetry={() => handleGenerate()}
        />
      )}

      {/* State 1: Setup & Topic Input */}
      {!quiz && (
        <Card className="p-6 sm:p-8 space-y-6">
          <div className="space-y-4">
            <Input
              label="What syllabus topic would you like to be quizzed on?"
              placeholder="e.g. Newton's Laws of Motion, Linear Algebra, Cell Structure..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              required
            />

            <div>
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 block mb-2">
                Number of Questions
              </label>
              <div className="flex gap-2">
                {[3, 5, 10].map((num) => (
                  <button
                    key={num}
                    type="button"
                    onClick={() => setNumQuestions(num)}
                    className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${numQuestions === num ? 'bg-indigo-600 text-white shadow-sm' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'}`}
                  >
                    {num} Questions
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Topic Pills */}
            <div>
              <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 block mb-2">
                Or pick a popular topic:
              </label>
              <div className="flex flex-wrap gap-2">
                {sampleTopics.map((t, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setTopic(t);
                      handleGenerate(t);
                    }}
                    className="text-xs px-3 py-1.5 rounded-xl bg-slate-50 dark:bg-slate-800/80 hover:bg-indigo-50 dark:hover:bg-indigo-950/40 border border-slate-200/80 dark:border-slate-700/80 hover:border-indigo-300 dark:hover:border-indigo-800 text-slate-700 dark:text-slate-300 transition-colors"
                  >
                    + {t}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
            <Button
              onClick={() => handleGenerate()}
              disabled={!topic.trim() || loading}
              loading={loading}
              size="lg"
              className="w-full sm:w-auto"
              icon={Sparkles}
            >
              Generate Practice Quiz
            </Button>
          </div>
        </Card>
      )}

      {/* State 2: Quiz Active / Answering */}
      {quiz && !result && (
        <Card className="p-6 sm:p-8 space-y-8">
          {/* Header & Progress */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                  Topic: {quiz.topic}
                </h2>
                <Badge variant="primary" size="sm">{quiz.questions.length} Questions</Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Answered {answeredCount} of {quiz.questions.length}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              icon={RotateCcw}
              onClick={() => { setQuiz(null); setResult(null); }}
            >
              Change Topic
            </Button>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
            <div
              className="bg-indigo-600 h-full transition-all duration-300 rounded-full"
              style={{ width: `${(answeredCount / quiz.questions.length) * 100}%` }}
            />
          </div>

          {/* Questions */}
          <div className="space-y-8">
            {quiz.questions.map((q, qIdx) => {
              const selectedOpt = userAnswers[qIdx];
              return (
                <div key={qIdx} className="space-y-3 p-4 sm:p-5 rounded-2xl bg-slate-50/60 dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60">
                  <div className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-lg bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
                      {qIdx + 1}
                    </span>
                    <h3 className="text-sm sm:text-base font-semibold text-slate-900 dark:text-white leading-snug">
                      {q.question}
                    </h3>
                  </div>

                  {/* Options */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-2">
                    {q.options.map((opt, oIdx) => {
                      const isChosen = selectedOpt === oIdx;
                      const letter = String.fromCharCode(65 + oIdx);
                      return (
                        <button
                          key={oIdx}
                          type="button"
                          onClick={() => handleSelectOption(qIdx, oIdx)}
                          className={`
                            flex items-center gap-3 p-3.5 rounded-xl border text-left text-xs sm:text-sm font-medium transition-all
                            ${isChosen 
                              ? 'bg-indigo-50 dark:bg-indigo-950/70 border-indigo-500 text-indigo-900 dark:text-indigo-200 shadow-xs' 
                              : 'bg-white dark:bg-slate-900 border-slate-200/80 dark:border-slate-800/80 text-slate-700 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-700'}
                          `}
                        >
                          <span className={`w-5 h-5 rounded-md font-bold text-xs flex items-center justify-center shrink-0 ${isChosen ? 'bg-indigo-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-500'}`}>
                            {letter}
                          </span>
                          <span className="flex-1">{opt}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="pt-4 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
            <span className="text-xs text-slate-400">
              {isComplete ? 'All questions answered!' : `Please answer remaining ${quiz.questions.length - answeredCount} questions.`}
            </span>
            <Button
              onClick={handleSubmit}
              disabled={!isComplete || submitting}
              loading={submitting}
              size="lg"
              icon={CheckSquare}
            >
              Submit Quiz for Evaluation
            </Button>
          </div>
        </Card>
      )}

      {/* State 3: Quiz Results & Score Breakdown */}
      {result && (
        <Card className="p-6 sm:p-8 space-y-6">
          {/* Result Score Banner */}
          <div className="p-6 rounded-2xl bg-gradient-to-tr from-indigo-50 via-indigo-100/50 to-purple-50 dark:from-indigo-950/50 dark:via-indigo-900/30 dark:to-purple-950/40 border border-indigo-200 dark:border-indigo-800/60 text-center space-y-3">
            <div className="w-14 h-14 mx-auto rounded-2xl bg-indigo-600 text-white flex items-center justify-center shadow-md shadow-indigo-600/20">
              <Award size={30} />
            </div>
            <h2 className="text-2xl font-black text-slate-900 dark:text-white">
              Score: {result.correct} / {result.total} ({( (result.correct / result.total) * 100 ).toFixed(0)}%)
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-300 max-w-md mx-auto">
              {result.correct === result.total 
                ? "Outstanding work! You've mastered this topic." 
                : result.correct >= result.total / 2 
                  ? "Good effort! Review the explanations below to refine your understanding."
                  : "Keep practicing! Check the detailed answers below to learn key concepts."}
            </p>
            <div className="flex justify-center gap-3 pt-2">
              <Button
                size="sm"
                onClick={() => { setResult(null); setUserAnswers(Array(quiz.questions.length).fill(-1)); }}
                icon={RotateCcw}
              >
                Retake Quiz
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => { setQuiz(null); setResult(null); setTopic(''); }}
              >
                New Topic
              </Button>
            </div>
          </div>

          {/* Detailed Question Review */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">
              Answer Analysis & Explanations:
            </h3>

            {quiz.questions.map((q, idx) => {
              const resItem = result.results?.[idx];
              const isCorrect = resItem ? resItem.correct : (userAnswers[idx] === (q.correct_answer ?? 0));
              const chosenOpt = userAnswers[idx];

              return (
                <div
                  key={idx}
                  className={`
                    p-5 rounded-2xl border text-xs text-left space-y-3
                    ${isCorrect 
                      ? 'bg-emerald-50/40 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/40' 
                      : 'bg-rose-50/40 dark:bg-rose-950/20 border-rose-200 dark:border-rose-900/40'}
                  `}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      {isCorrect ? (
                        <CheckCircle2 size={18} className="text-emerald-600 dark:text-emerald-400 shrink-0" />
                      ) : (
                        <XCircle size={18} className="text-rose-600 dark:text-rose-400 shrink-0" />
                      )}
                      <h4 className="font-bold text-slate-900 dark:text-white text-sm">
                        Q{idx + 1}: {q.question}
                      </h4>
                    </div>
                    <Badge variant={isCorrect ? 'success' : 'danger'} size="sm">
                      {isCorrect ? 'Correct' : 'Incorrect'}
                    </Badge>
                  </div>

                  <div className="space-y-1.5 pl-6 text-slate-700 dark:text-slate-300">
                    <p>
                      <strong>Your Answer:</strong> {chosenOpt !== -1 ? `${String.fromCharCode(65 + chosenOpt)}. ${q.options[chosenOpt]}` : 'Unanswered'}
                    </p>
                    {resItem?.explanation && (
                      <div className="mt-2 p-3 rounded-xl bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/60 text-slate-600 dark:text-slate-300">
                        <strong className="text-indigo-600 dark:text-indigo-400">Explanation: </strong>
                        {resItem.explanation}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
};
