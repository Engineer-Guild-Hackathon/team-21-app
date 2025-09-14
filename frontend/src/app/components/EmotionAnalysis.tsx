'use client';

import {
  ChartBarIcon,
  ExclamationTriangleIcon,
  FaceSmileIcon,
  FireIcon,
  HeartIcon,
} from '@heroicons/react/24/outline';
import { useState } from 'react';

interface EmotionData {
  emotion: string;
  score: number;
  color: string;
  icon: any;
  description: string;
}

export default function EmotionAnalysis() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [emotions, setEmotions] = useState<EmotionData[]>([]);
  const [showTips, setShowTips] = useState(false);

  const emotionMap: { [key: string]: EmotionData } = {
    joy: {
      emotion: '喜び',
      score: 0,
      color: 'bg-yellow-100 text-yellow-800',
      icon: FaceSmileIcon,
      description: '前向きで活力のある状態です。この感情を活かして新しい課題に挑戦してみましょう。',
    },
    frustration: {
      emotion: 'フラストレーション',
      score: 0,
      color: 'bg-red-100 text-red-800',
      icon: ExclamationTriangleIcon,
      description: '困難を感じている状態です。一度深呼吸して、問題を小さく分解してみましょう。',
    },
    motivation: {
      emotion: 'モチベーション',
      score: 0,
      color: 'bg-green-100 text-green-800',
      icon: FireIcon,
      description: 'やる気に満ちた状態です。目標を設定して、計画的に進めていきましょう。',
    },
    concentration: {
      emotion: '集中',
      score: 0,
      color: 'bg-blue-100 text-blue-800',
      icon: ChartBarIcon,
      description: '集中力が高まっている状態です。この状態を維持して課題に取り組みましょう。',
    },
  };

  const analyzeEmotion = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      // モックデータを使用（実際のAPIが実装されるまでの仮実装）
      await new Promise(resolve => setTimeout(resolve, 1000)); // 1秒の遅延をシミュレート

      // テキストの内容に基づいてランダムな感情スコアを生成
      const mockEmotions = {
        joy: Math.random() * 0.8 + 0.2,
        frustration: Math.random() * 0.6 + 0.1,
        motivation: Math.random() * 0.7 + 0.3,
        concentration: Math.random() * 0.6 + 0.2,
      };

      const updatedEmotions = Object.entries(mockEmotions).map(([key, score]) => ({
        ...emotionMap[key],
        score: score as number,
      }));

      setEmotions(updatedEmotions);
      setShowTips(true);
    } catch (error) {
      console.error('Error:', error);
      alert('感情分析中にエラーが発生しました。もう一度お試しください。');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 入力フォーム */}
      <div className="space-y-4">
        <div className="flex flex-col space-y-2">
          <label htmlFor="emotion-text" className="text-sm font-medium text-gray-700">
            あなたの気持ちを教えてください
          </label>
          <textarea
            id="emotion-text"
            rows={4}
            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            placeholder="今の気持ちや状況を自由に書いてください..."
            value={text}
            onChange={e => setText(e.target.value)}
          />
        </div>
        <button
          onClick={analyzeEmotion}
          disabled={loading || !text.trim()}
          className={`w-full rounded-md px-4 py-2 text-sm font-semibold text-white shadow-sm ${
            loading || !text.trim()
              ? 'bg-gray-300 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-500'
          }`}
        >
          {loading ? '分析中...' : '感情を分析する'}
        </button>
      </div>

      {/* 結果表示 */}
      {emotions.length > 0 && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {emotions.map(emotion => (
              <div
                key={emotion.emotion}
                className={`rounded-lg p-4 ${emotion.color} flex items-start space-x-4`}
              >
                <emotion.icon className="h-6 w-6 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold">{emotion.emotion}</h3>
                  <div className="mt-1">
                    <div className="w-full bg-white rounded-full h-2">
                      <div
                        className="bg-current h-2 rounded-full transition-all duration-500"
                        style={{ width: `${emotion.score * 100}%` }}
                      />
                    </div>
                    <p className="mt-2 text-sm">{emotion.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* アドバイスセクション */}
          {showTips && (
            <div className="mt-6 bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                <HeartIcon className="h-6 w-6 inline-block mr-2 text-pink-500" />
                あなたへのアドバイス
              </h3>
              <div className="space-y-4">
                {emotions
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 2)
                  .map(emotion => (
                    <div key={`tip-${emotion.emotion}`} className="text-gray-600">
                      <p className="mb-2">
                        <span className="font-medium">{emotion.emotion}</span>
                        の感情が強く表れています。
                      </p>
                      <p>{emotion.description}</p>
                    </div>
                  ))}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-sm text-gray-500">
                    これらの感情を認識することは、学習効果を高める第一歩です。
                    自分の感情に気づき、それを活かして学習を進めていきましょう。
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
