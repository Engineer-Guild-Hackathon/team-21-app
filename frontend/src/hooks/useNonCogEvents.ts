import { useCallback } from 'react';

export type LearnAction = 'answer_submit' | 'hint_request' | 'retry' | 'give_up';
export type Difficulty = 'easy' | 'normal' | 'challenge';

export interface LearnActionEvent {
  event_id: string;
  user_id: string;
  session_id: string;
  action: LearnAction;
  think_time_ms: number;
  success?: boolean;
  difficulty?: Difficulty;
  created_at?: string; // ISO8601
}

export interface NonCogSummary {
  user_id: string;
  retry_count: number;
  avg_think_time_ms: number;
  re_challenge_rate: number;
  grit_score: number;
  srl_score: number;
  updated_at: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export function useNonCogEvents() {
  const postEvent = useCallback(async (payload: LearnActionEvent) => {
    const res = await fetch(`${API_BASE}/api/learning/events/learn-action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`Failed to post event: ${res.status}`);
    }
    return (await res.json()) as { status: string };
  }, []);

  const fetchSummary = useCallback(async (userId: string) => {
    const res = await fetch(
      `${API_BASE}/api/learning/metrics/noncog-summary?user_id=${encodeURIComponent(userId)}`
    );
    if (!res.ok) {
      throw new Error(`Failed to fetch summary: ${res.status}`);
    }
    return (await res.json()) as NonCogSummary;
  }, []);

  return { postEvent, fetchSummary };
}
