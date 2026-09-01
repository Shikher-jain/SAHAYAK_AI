import React, { useState } from 'react';
import { Bot, User, Lock, Mail, ArrowRight, Sparkles, CheckCircle2, Eye, EyeOff } from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Card } from '../components/ui/Card';

export const AuthPage = () => {
  const { login, showSuccess, showError } = useAppContext();
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    email: '',
    fullname: '',
    role: 'student'
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const validate = () => {
    const newErrors = {};
    if (!formData.username.trim()) newErrors.username = 'Username is required';
    if (!formData.password) newErrors.password = 'Password is required';
    else if (formData.password.length < 4) newErrors.password = 'Password must be at least 4 characters';

    if (!isLogin) {
      if (!formData.email.trim()) newErrors.email = 'Email is required';
      else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Invalid email address';
      if (!formData.fullname.trim()) newErrors.fullname = 'Full name is required';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    setLoading(true);
    setErrors({});

    const endpoint = isLogin ? "/auth/login" : "/auth/register";
    const payload = isLogin 
      ? { username: formData.username.trim(), password: formData.password }
      : { 
          username: formData.username.trim(), 
          email: formData.email.trim(), 
          password: formData.password, 
          full_name: formData.fullname.trim(), 
          role: formData.role 
        };

    const { ok, data, error } = await callBackend('post', endpoint, payload);

    if (ok && data) {
      if (isLogin) {
        login(data.access_token, data.username || formData.username, data.role || 'student');
      } else {
        showSuccess('Account created successfully! Please sign in.');
        setIsLogin(true);
      }
    } else {
      showError(error || 'Authentication request failed');
      setErrors({ form: error || 'Authentication failed. Please check your credentials.' });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-[#090d16] p-4 sm:p-6 lg:p-8 font-sans">
      <div className="w-full max-w-4xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        {/* Left Hero / Brand section */}
        <div className="lg:col-span-6 space-y-6 text-left hidden lg:block pr-4">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200/60 dark:border-indigo-800/60 text-indigo-700 dark:text-indigo-300 text-xs font-semibold">
            <Sparkles size={14} className="text-indigo-600" />
            Next-Gen Multimodal Learning Platform
          </div>

          <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight">
            Learn smarter with <span className="text-indigo-600 dark:text-indigo-400">Sahayak AI</span>.
          </h1>

          <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed">
            Ingest study materials, generate adaptive quizzes, visualize knowledge graphs, and consult with personalized AI mentors in your preferred language.
          </p>

          <div className="space-y-3 pt-2">
            {[
              "Multimodal Document Ingestion (PDFs, URLs, Notes)",
              "Semantic Vector & RAG Knowledge Retrieval",
              "Adaptive Quizzing & Real-time Progress Tracking",
              "Multilingual Chat & Domain AI Mentorship"
            ].map((feat, idx) => (
              <div key={idx} className="flex items-center gap-2.5 text-xs font-medium text-slate-700 dark:text-slate-300">
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
                <span>{feat}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right Form Card */}
        <div className="lg:col-span-6 w-full max-w-md mx-auto">
          <Card className="shadow-xl border-slate-200/90 dark:border-slate-800/90 p-8">
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white mb-3 shadow-md shadow-indigo-500/20">
                <Bot size={26} />
              </div>
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                {isLogin ? 'Sign in to Sahayak' : 'Create an Account'}
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                {isLogin ? 'Enter your credentials to continue learning' : 'Join Sahayak to access your AI companion'}
              </p>
            </div>

            {/* Login / Register Toggle Tabs */}
            <div className="flex p-1 bg-slate-100 dark:bg-slate-800 rounded-xl mb-6">
              <button
                type="button"
                onClick={() => { setIsLogin(true); setErrors({}); }}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${isLogin ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm' : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-100'}`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => { setIsLogin(false); setErrors({}); }}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${!isLogin ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm' : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-100'}`}
              >
                Register
              </button>
            </div>

            {errors.form && (
              <div className="mb-4 p-3 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 rounded-xl text-xs text-rose-600 dark:text-rose-400 text-left">
                {errors.form}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4 text-left">
              <Input
                label="Username"
                placeholder="e.g. shikher_ai"
                icon={User}
                required
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                error={errors.username}
              />

              {!isLogin && (
                <>
                  <Input
                    label="Full Name"
                    placeholder="e.g. Shikher Jain"
                    icon={User}
                    required
                    value={formData.fullname}
                    onChange={(e) => setFormData({ ...formData, fullname: e.target.value })}
                    error={errors.fullname}
                  />

                  <Input
                    label="Email Address"
                    type="email"
                    placeholder="e.g. student@sahayak.ai"
                    icon={Mail}
                    required
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    error={errors.email}
                  />

                  <Select
                    label="Learning Role"
                    value={formData.role}
                    onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                    options={[
                      { value: 'student', label: 'Student (Learner)' },
                      { value: 'teacher', label: 'Teacher / Educator' },
                      { value: 'admin', label: 'Administrator' }
                    ]}
                  />
                </>
              )}

              <Input
                label="Password"
                type={showPassword ? 'text' : 'password'}
                placeholder="••••••••"
                icon={Lock}
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                error={errors.password}
                rightElement={
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                }
              />

              <Button
                type="submit"
                className="w-full mt-2"
                size="lg"
                loading={loading}
                icon={ArrowRight}
                iconPosition="right"
              >
                {isLogin ? 'Sign In' : 'Create Account'}
              </Button>
            </form>

            <div className="mt-6 pt-4 border-t border-slate-100 dark:border-slate-800 text-center">
              <p className="text-xs text-slate-400">
                {isLogin ? "Don't have an account? " : "Already have an account? "}
                <button
                  type="button"
                  onClick={() => { setIsLogin(!isLogin); setErrors({}); }}
                  className="font-bold text-indigo-600 dark:text-indigo-400 hover:underline"
                >
                  {isLogin ? 'Create one now' : 'Sign in'}
                </button>
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
