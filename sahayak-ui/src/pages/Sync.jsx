import React, { useState, useEffect } from 'react';
import { RefreshCcw, Download, Server, CloudOff } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { PageHeader } from '../components/ui/PageHeader';
import { Button } from '../components/ui/Button';
import { callBackend } from '../api/client';
import { useAppContext } from '../context/AppContext';

export const Sync = () => {
  const { showError, showSuccess } = useAppContext();
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    setLoading(true);
    const { ok, data, error } = await callBackend('get', '/sync/status');
    if (ok && data) {
      setStatus(data);
    } else {
      showError(error || 'Failed to fetch sync status');
    }
    setLoading(false);
  };

  const handleExport = async () => {
    setExporting(true);
    const { ok, data, error } = await callBackend('post', '/sync/export');
    if (ok && data) {
      showSuccess('Data exported successfully for offline use');
      // If data contains a download URL or raw data, handle it:
      if (data.download_url) {
        window.open(data.download_url, '_blank');
      } else if (data.data) {
        // Create a blob and download it
        const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sahayak_offline_export.json';
        a.click();
        URL.revokeObjectURL(url);
      }
    } else {
      showError(error || 'Export failed');
    }
    setExporting(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <PageHeader 
        title="Data Sync & Offline Mode" 
        subtitle="Manage your local data synchronization and offline access"
        icon={RefreshCcw} 
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="space-y-4">
          <div className="flex items-center gap-3">
            <div className={`p-3 rounded-xl ${status?.is_online ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400' : 'bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400'}`}>
              {status?.is_online ? <Server size={24} /> : <CloudOff size={24} />}
            </div>
            <div>
              <h3 className="font-bold text-slate-800 dark:text-slate-200">Connection Status</h3>
              <p className="text-sm text-slate-500">
                {loading ? 'Checking...' : status?.is_online ? 'Connected to Cloud' : 'Offline / Local Mode'}
              </p>
            </div>
          </div>
          
          <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
            <h4 className="text-sm font-semibold mb-2">Sync Stats</h4>
            {loading ? (
              <div className="animate-pulse space-y-2">
                <div className="h-4 bg-slate-200 dark:bg-slate-800 w-1/2 rounded" />
                <div className="h-4 bg-slate-200 dark:bg-slate-800 w-2/3 rounded" />
              </div>
            ) : (
              <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                <li>Last Synced: {status?.last_sync || 'Never'}</li>
                <li>Pending Changes: {status?.pending_changes || 0}</li>
                <li>Local Database Size: {status?.local_size || 'Unknown'}</li>
              </ul>
            )}
          </div>

          <Button 
            onClick={fetchStatus} 
            loading={loading}
            variant="outline" 
            className="w-full mt-4"
          >
            Refresh Status
          </Button>
        </Card>

        <Card className="space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-indigo-100 dark:bg-indigo-900/50 text-indigo-600 dark:text-indigo-400">
                <Download size={24} />
              </div>
              <div>
                <h3 className="font-bold text-slate-800 dark:text-slate-200">Offline Export</h3>
                <p className="text-sm text-slate-500">
                  Download your data for offline access
                </p>
              </div>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Export your knowledge base, notes, and profile data to a local file. You can use this file with the Sahayak Desktop App when you don't have internet access.
            </p>
          </div>

          <Button 
            onClick={handleExport}
            loading={exporting}
            icon={Download}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white"
          >
            {exporting ? 'Preparing Export...' : 'Export Data Now'}
          </Button>
        </Card>
      </div>
    </div>
  );
};
