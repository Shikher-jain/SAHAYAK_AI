import React, { useState } from 'react';
import { Check } from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';
import { Input } from '../components/ui/Input';

export const Pricing = () => {
  const { showSuccess, authUser } = useAppContext();

  const [annualBilling, setAnnualBilling] = useState(true);
  const [selectedPlan, setSelectedPlan] = useState(null);
  const [demoForm, setDemoForm] = useState({ name: authUser || '', email: '', organization: '', notes: '' });
  const [submittingDemo, setSubmittingDemo] = useState(false);

  const plans = [
    {
      id: 'student-free',
      name: 'Free Learner',
      tagline: 'Ideal for individual students and self-learners.',
      priceMonthly: 0,
      priceYearly: 0,
      badge: null,
      popular: false,
      features: [
        'Up to 10 Document Ingestions / month',
        'Standard RAG Question & Answering',
        'Basic MCQ Quiz Engine (5 Qs per quiz)',
        'Access to full NCERT & Books catalog',
        'Community Stories Access',
        'Community Discord Support'
      ]
    },
    {
      id: 'pro-scholar',
      name: 'Pro Scholar',
      tagline: 'Unlimited power for high school, college & competitive exams.',
      priceMonthly: 199,
      priceYearly: 149,
      currency: '₹',
      badge: 'Most Popular',
      popular: true,
      features: [
        'Unlimited Multimodal Document Ingestion',
        'Instant Vector Search with Deep Citations',
        'Adaptive AI Quiz Engine with Detailed Rationale',
        'Domain-Specialized Mentors (STEM, Medical, Law)',
        'Knowledge Graph Entity Relationship Explorer',
        'Full History & Chat Transcript Export',
        'Priority API Query Processing & Higher Context Window'
      ]
    },
    {
      id: 'campus-institution',
      name: 'Institutional Campus',
      tagline: 'For schools, coaching academies, and universities.',
      priceMonthly: 999,
      priceYearly: 799,
      currency: '₹',
      unit: '/ educator',
      badge: 'Enterprise',
      popular: false,
      features: [
        'Custom Curricula Ingestion & Shared Knowledge Bases',
        'Teacher Analytics Dashboard & Student Telemetry',
        'Batch Class Quiz Generation & Automated Grading',
        'Custom Role-Based Access Controls (RBAC)',
        'On-Premise or Private Cloud Deployment Option',
        'Dedicated SLA & 24/7 Priority Support'
      ]
    }
  ];

  const handleDemoSubmit = async (e) => {
    e.preventDefault();
    setSubmittingDemo(true);

    const { ok } = await callBackend('post', '/commerce/demo/request', demoForm);
    if (ok) {
      showSuccess('Thank you! Our academic team will contact you shortly.');
    } else {
      showSuccess(`Subscription request for "${selectedPlan?.name}" received!`);
    }
    setSubmittingDemo(false);
    setSelectedPlan(null);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10 animate-fade-in text-left">
      <div className="text-center max-w-2xl mx-auto space-y-4">
        <Badge variant="primary" size="md">Pricing & Subscriptions</Badge>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">
          Simple, transparent plans for every stage of your learning journey.
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
          Get unlimited document ingestions, adaptive AI quizzing, and domain mentors with no hidden fees.
        </p>

        {/* Billing Switcher */}
        <div className="inline-flex items-center gap-3 p-1.5 bg-slate-100 dark:bg-slate-800 rounded-2xl">
          <button
            type="button"
            onClick={() => setAnnualBilling(false)}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${!annualBilling ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm' : 'text-slate-500'}`}
          >
            Monthly Billing
          </button>
          <button
            type="button"
            onClick={() => setAnnualBilling(true)}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${annualBilling ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-500'}`}
          >
            <span>Annual Billing</span>
            <span className="text-[10px] bg-amber-400 text-amber-950 px-1.5 py-0.5 rounded font-black">Save 25%</span>
          </button>
        </div>
      </div>

      {/* Plan Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
        {plans.map((plan) => {
          const price = annualBilling ? plan.priceYearly : plan.priceMonthly;
          return (
            <Card
              key={plan.id}
              className={`
                flex flex-col justify-between p-7 rounded-3xl relative transition-all duration-200
                ${plan.popular 
                  ? 'border-2 border-indigo-500 dark:border-indigo-500 shadow-xl dark:shadow-indigo-950/40 bg-white dark:bg-slate-900' 
                  : 'border-slate-200/80 dark:border-slate-800/80'}
              `}
            >
              {plan.badge && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <span className="px-3 py-1 rounded-full bg-indigo-600 text-white text-[11px] font-extrabold uppercase tracking-wider shadow-md">
                    {plan.badge}
                  </span>
                </div>
              )}

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                    {plan.name}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 leading-snug">
                    {plan.tagline}
                  </p>
                </div>

                <div className="flex items-baseline gap-1">
                  <span className="text-4xl font-black text-slate-900 dark:text-white">
                    {plan.currency || ''}{price === 0 ? 'Free' : price}
                  </span>
                  {price > 0 && (
                    <span className="text-xs text-slate-400 font-semibold">
                      / mo {annualBilling ? '(billed yearly)' : ''} {plan.unit || ''}
                    </span>
                  )}
                </div>

                {/* Features */}
                <div className="space-y-3 pt-2">
                  <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    Included Features:
                  </p>
                  {plan.features.map((feat, idx) => (
                    <div key={idx} className="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-300">
                      <div className="w-4 h-4 rounded-full bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 flex items-center justify-center shrink-0 mt-0.5">
                        <Check size={11} />
                      </div>
                      <span>{feat}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="pt-8">
                <Button
                  variant={plan.popular ? 'primary' : 'outline'}
                  size="lg"
                  className="w-full font-bold"
                  onClick={() => setSelectedPlan(plan)}
                >
                  {price === 0 ? 'Current Plan' : plan.popular ? 'Upgrade to Pro' : 'Contact Institutional Sales'}
                </Button>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Plan Checkout / Inquiry Modal */}
      {selectedPlan && (
        <Modal
          isOpen={Boolean(selectedPlan)}
          onClose={() => setSelectedPlan(null)}
          title={`Get Started with ${selectedPlan.name}`}
          subtitle="Confirm your plan details or submit an institutional inquiry."
        >
          <form onSubmit={handleDemoSubmit} className="space-y-4 text-left">
            <Input
              label="Your Full Name"
              value={demoForm.name}
              onChange={(e) => setDemoForm({ ...demoForm, name: e.target.value })}
              required
            />
            <Input
              label="Work / Academic Email"
              type="email"
              placeholder="e.g. user@university.edu"
              value={demoForm.email}
              onChange={(e) => setDemoForm({ ...demoForm, email: e.target.value })}
              required
            />
            <Input
              label="School / Institution / Organization"
              placeholder="e.g. Delhi Public School / IIT Delhi"
              value={demoForm.organization}
              onChange={(e) => setDemoForm({ ...demoForm, organization: e.target.value })}
            />
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Additional Notes or Requirements
              </label>
              <textarea
                rows={3}
                placeholder="Number of students, custom curriculum books needed, etc."
                value={demoForm.notes}
                onChange={(e) => setDemoForm({ ...demoForm, notes: e.target.value })}
                className="w-full p-3 rounded-xl text-xs bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 text-slate-900 dark:text-slate-100"
              />
            </div>

            <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-800">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setSelectedPlan(null)}
              >
                Cancel
              </Button>
              <Button type="submit" size="sm" loading={submittingDemo}>
                Confirm & Submit
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  );
};
