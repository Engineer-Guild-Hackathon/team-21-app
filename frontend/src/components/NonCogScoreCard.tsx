import React from 'react';
import type { NonCogSummary } from '../hooks/useNonCogEvents';

type Props = {
  summary: NonCogSummary | null;
  loading?: boolean;
  error?: string | null;
};

export const NonCogScoreCard: React.FC<Props> = ({ summary, loading, error }) => {
  if (loading) return <div className="p-4 rounded border">Loading...</div>;
  if (error) return <div className="p-4 rounded border text-red-600">{error}</div>;
  if (!summary) return <div className="p-4 rounded border">No data</div>;

  return (
    <div className="p-4 rounded border space-y-2">
      <div className="font-semibold">User: {summary.user_id}</div>
      <div className="grid grid-cols-2 gap-2">
        <div>Retry Count: {summary.retry_count}</div>
        <div>Avg Think: {Math.round(summary.avg_think_time_ms)} ms</div>
        <div>Re-challenge Rate: {(summary.re_challenge_rate * 100).toFixed(1)}%</div>
        <div>Grit: {summary.grit_score}</div>
        <div>SRL: {summary.srl_score}</div>
      </div>
      <div className="text-xs text-gray-500">
        Updated: {new Date(summary.updated_at).toLocaleString()}
      </div>
    </div>
  );
};
