'use client';

import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface EmotionAnalysisResult {
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
    frustration: number;
    concentration: number;
  };
  feedback: string;
  next_action: {
    type: string;
    description: string;
  };
}

export default function EmotionAnalysis() {
  const { user } = useAuth();
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmotionAnalysisResult | null>(null);
  const [error, setError] = useState('');

  const analyzeEmotion = async () => {
    if (!text.trim()) {
      setError('テキストを入力してください');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/emotion-analysis/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          text,
          context: 'text_input',
        }),
      });

      if (!response.ok) {
        throw new Error('感情分析に失敗しました');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '感情分析中にエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  const renderEmotionBars = () => {
    if (!result) return null;

    return (
      <div className="space-y-2">
        {Object.entries(result.emotions).map(([emotion, value]) => (
          <div key={emotion} className="flex items-center">
            <span className="w-24 text-sm">{emotion}</span>
            <div className="flex-1 h-4 bg-gray-200 rounded">
              <div className="h-full bg-blue-500 rounded" style={{ width: `${value * 100}%` }} />
            </div>
            <span className="w-16 text-right text-sm">{(value * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6 p-6 bg-white rounded-lg shadow">
      <div>
        <label htmlFor="emotion-text" className="block text-sm font-medium text-gray-700">
          あなたの気持ちを教えてください
        </label>
        <div className="mt-1">
          <textarea
            id="emotion-text"
            rows={4}
            className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md"
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="今の気持ちや状況を入力してください..."
          />
        </div>
      </div>

      {error && <div className="text-red-600 text-sm">{error}</div>}

      <button
        onClick={analyzeEmotion}
        disabled={loading}
        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
      >
        {loading ? '分析中...' : '感情を分析'}
      </button>

      {result && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-medium text-gray-900">感情分析結果</h3>
            {renderEmotionBars()}
          </div>

          <div>
            <h3 className="text-lg font-medium text-gray-900">フィードバック</h3>
            <p className="mt-1 text-sm text-gray-600">{result.feedback}</p>
          </div>

          <div>
            <h3 className="text-lg font-medium text-gray-900">次のアクション</h3>
            <p className="mt-1 text-sm text-gray-600">{result.next_action.description}</p>
          </div>
        </div>
      )}
    </div>
  );
}
