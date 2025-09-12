import { useEffect, useRef, useState } from 'react';

interface SSEMessage {
  type: string;
  user_id?: string;
  retry_count?: number;
  avg_think_time_ms?: number;
  re_challenge_rate?: number;
  grit_score?: number;
  srl_score?: number;
  updated_at?: string;
}

interface UseSSEOptions {
  userId: string;
  onMessage?: (message: SSEMessage) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export const useSSE = ({ userId, onMessage, onError, onOpen, onClose }: UseSSEOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<SSEMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  // 最新のコールバックを参照に保持（再接続を避ける）
  const onMessageRef = useRef<typeof onMessage>();
  const onErrorRef = useRef<typeof onError>();
  const onOpenRef = useRef<typeof onOpen>();
  const onCloseRef = useRef<typeof onClose>();

  useEffect(() => {
    onMessageRef.current = onMessage;
    onErrorRef.current = onError;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
  }, [onMessage, onError, onOpen, onClose]);

  useEffect(() => {
    if (!userId) return;

    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
    const url = `${API_BASE}/api/learning/stream/${userId}`;

    console.log('SSE: Connecting to', url);

    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log('SSE: Connection opened');
      setIsConnected(true);
      setError(null);
      onOpenRef.current?.();
    };

    eventSource.onmessage = event => {
      try {
        const data = JSON.parse(event.data);
        console.log('SSE: Message received', data);
        setLastMessage(data);
        onMessageRef.current?.(data);
      } catch (err) {
        console.error('SSE: Failed to parse message', err);
        setError('Failed to parse message');
      }
    };

    eventSource.onerror = error => {
      console.error('SSE: Connection error', error);
      setIsConnected(false);
      setError('Connection error');
      onErrorRef.current?.(error);
    };

    return () => {
      console.log('SSE: Disconnecting');
      eventSource.close();
      setIsConnected(false);
      onCloseRef.current?.();
    };
  }, [userId]);

  const disconnect = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    }
  };

  return {
    isConnected,
    lastMessage,
    error,
    disconnect,
  };
};
