import React, { useState, useEffect } from 'react';
import { Mail, Phone, MapPin, Send } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { PageHeader } from '../components/ui/PageHeader';
import { callBackend } from '../api/client';
import { useAppContext } from '../context/AppContext';

export const Contact = () => {
  const { showError } = useAppContext();
  const [contactInfo, setContactInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchContactInfo();
  }, []);

  const fetchContactInfo = async () => {
    setLoading(true);
    const { ok, data, error } = await callBackend('get', '/pages/contact');
    if (ok && data) {
      setContactInfo(data);
    } else {
      showError(error || 'Failed to fetch contact information');
      // Fallback
      setContactInfo({
        email: 'info@sahayakai.com',
        support_email: 'support@sahayakai.com'
      });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader 
        title="Contact Us" 
        subtitle="We'd love to hear from you. Get in touch with the Sahayak AI team."
        icon={Mail} 
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Contact Information */}
        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200 mb-6">Contact Information</h3>
            
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 rounded-xl">
                  <Mail size={20} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">General Inquiries</h4>
                  {loading ? (
                    <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-32 mt-1 animate-pulse" />
                  ) : (
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">{contactInfo?.email}</p>
                  )}
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="p-3 bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 rounded-xl">
                  <Send size={20} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Support</h4>
                  {loading ? (
                    <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-32 mt-1 animate-pulse" />
                  ) : (
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">{contactInfo?.support_email}</p>
                  )}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Contact Form */}
        <Card className="p-6">
          <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200 mb-6">Send a Message</h3>
          <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); alert('Message Sent!'); }}>
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Your Name</label>
              <input type="text" className="w-full p-3 rounded-xl text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="John Doe" required />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Your Email</label>
              <input type="email" className="w-full p-3 rounded-xl text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="john@example.com" required />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Message</label>
              <textarea rows={4} className="w-full p-3 rounded-xl text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="How can we help?" required />
            </div>
            <button type="submit" className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-bold rounded-xl transition-colors">
              Send Message
            </button>
          </form>
        </Card>
      </div>
    </div>
  );
};
