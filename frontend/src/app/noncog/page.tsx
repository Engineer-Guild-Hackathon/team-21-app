'use client';

import { useEffect, useState } from 'react';
import { NonCogScoreCard } from '../../components/NonCogScoreCard';
import { useNonCogEvents, type NonCogSummary } from '../../hooks/useNonCogEvents';
import { useSSE } from '../../hooks/useSSE';

export default function NonCogPage() {
  const { postEvent, fetchSummary } = useNonCogEvents();
  const [summary, setSummary] = useState<NonCogSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const userId = '1';

  // SSEでリアルタイム更新を受信
  const { isConnected } = useSSE({
    userId,
    onMessage: message => {
      if (message.type === 'summary_update') {
        console.log('Real-time update received:', message);
        setSummary({
          user_id: message.user_id || userId,
          retry_count: message.retry_count || 0,
          avg_think_time_ms: message.avg_think_time_ms || 0,
          re_challenge_rate: message.re_challenge_rate || 0,
          grit_score: message.grit_score || 0,
          srl_score: message.srl_score || 0,
          updated_at: message.updated_at || new Date().toISOString(),
        });
      }
    },
    onError: error => {
      console.error('SSE Error:', error);
      setError('Real-time connection error');
    },
  });

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchSummary(userId);
      setSummary(data);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendRetry = async () => {
    try {
      await postEvent({
        event_id: `ui-${Date.now()}`,
        user_id: userId,
        session_id: 'sess-ui',
        action: 'retry',
        think_time_ms: Math.floor(Math.random() * 4000) + 500,
        success: true,
        difficulty: 'normal',
        created_at: new Date().toISOString(),
      });
      // リアルタイム更新があるので手動リフレッシュは不要
      console.log('Event sent, waiting for real-time update...');
    } catch (e: any) {
      setError(e?.message ?? 'Failed to send');
    }
  };

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-bold">Non-Cog Summary</h1>
      <div className="flex gap-2 items-center">
        <button
          onClick={load}
          className="px-3 py-2 rounded bg-blue-600 text-white hover:bg-blue-700"
        >
          Refresh Summary
        </button>
        <button
          onClick={sendRetry}
          className="px-3 py-2 rounded bg-emerald-600 text-white hover:bg-emerald-700"
        >
          Send Retry Event
        </button>
        <div
          className={`px-2 py-1 rounded text-sm ${
            isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}
        >
          {isConnected ? '🟢 Real-time Connected' : '🔴 Disconnected'}
        </div>
      </div>
      <NonCogScoreCard summary={summary} loading={loading} error={error} />
    </main>
  );
}
