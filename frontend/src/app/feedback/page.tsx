'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface MLFeedback {
  user_id: number;
  skills: {
    grit: number;
    collaboration: number;
    self_regulation: number;
    emotional_intelligence: number;
    confidence: number;
  };
  feedback: string;
  analysis_timestamp: string;
}

export default function FeedbackPage() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [mlFeedback, setMlFeedback] = useState<MLFeedback | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/auth/login?redirect=/feedback');
      return;
    }

    fetchMLFeedback();
  }, [isAuthenticated, router]);

  const fetchMLFeedback = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      // 最新のML分析結果を取得（実際の実装では専用APIエンドポイントが必要）
      const response = await fetch('http://localhost:8000/api/avatars/stats', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const stats = await response.json();

        // ダミーのML分析結果を生成（実際の実装では専用APIから取得）
        const mockFeedback: MLFeedback = {
          user_id: user?.id || 0,
          skills: {
            grit: stats.grit_level || 2.5,
            collaboration: stats.collaboration_level || 2.3,
            self_regulation: stats.self_regulation_level || 2.8,
            emotional_intelligence: stats.emotional_intelligence_level || 2.1,
            confidence: 2.4,
          },
          feedback: generateFeedbackFromSkills({
            grit: stats.grit_level || 2.5,
            collaboration: stats.collaboration_level || 2.3,
            self_regulation: stats.self_regulation_level || 2.8,
            emotional_intelligence: stats.emotional_intelligence_level || 2.1,
            confidence: 2.4,
          }),
          analysis_timestamp: new Date().toISOString(),
        };

        setMlFeedback(mockFeedback);
      }
    } catch (error) {
      console.error('フィードバック取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateFeedbackFromSkills = (skills: any) => {
    const feedbacks = [];

    if (skills.grit >= 4.0) {
      feedbacks.push(
        '🌟 素晴らしいやり抜く力を持っています！困難な課題にも諦めずに取り組む姿勢が見られます。'
      );
    } else if (skills.grit >= 3.0) {
      feedbacks.push('👍 やり抜く力が向上しています。目標を設定して継続的に取り組んでみましょう。');
    } else {
      feedbacks.push(
        '💪 やり抜く力を鍛えるために、小さな目標から始めて達成感を積み重ねていきましょう。'
      );
    }

    if (skills.collaboration >= 4.0) {
      feedbacks.push('🤝 協調性がとても高いです！他者との協力を大切にしていますね。');
    } else if (skills.collaboration >= 3.0) {
      feedbacks.push('👥 協調性が育っています。グループ学習やペア学習を活用してみましょう。');
    } else {
      feedbacks.push(
        '🤝 協調性を高めるために、友達と一緒に勉強したり、質問を積極的にしてみましょう。'
      );
    }

    return feedbacks.join('\n\n');
  };

  if (isLoading) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center">
        <div className="text-xl">フィードバックを読み込み中...</div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">学習フィードバック</h1>

      <div className="w-full max-w-7xl">
        {/* 最新のフィードバック */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">AI分析によるフィードバック</h2>
          <div className="space-y-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <p className="text-gray-600 whitespace-pre-line">
                {mlFeedback?.feedback ||
                  'まだ分析データがありません。AIチャットやクエストを利用してデータを蓄積してください。'}
              </p>
              <p className="text-sm text-gray-500 mt-2">
                {mlFeedback
                  ? new Date(mlFeedback.analysis_timestamp).toLocaleString('ja-JP')
                  : '分析待ち'}
              </p>
            </div>
          </div>
        </div>

        {/* スキル別フィードバック */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* コミュニケーションスキル */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">コミュニケーションスキル</h3>
            <div className="space-y-2">
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>積極的な発言</span>
              </div>
              <div className="flex items-center">
                <span className="text-yellow-500 mr-2">●</span>
                <span>傾聴スキル</span>
              </div>
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>非言語コミュニケーション</span>
              </div>
            </div>
          </div>

          {/* 感情コントロール */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">感情コントロール</h3>
            <div className="space-y-2">
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>感情認識</span>
              </div>
              <div className="flex items-center">
                <span className="text-yellow-500 mr-2">●</span>
                <span>ストレス管理</span>
              </div>
              <div className="flex items-center">
                <span className="text-red-500 mr-2">●</span>
                <span>感情表現</span>
              </div>
            </div>
          </div>

          {/* 目標設定とモチベーション */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">目標設定とモチベーション</h3>
            <div className="space-y-2">
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>目標設定</span>
              </div>
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>進捗管理</span>
              </div>
              <div className="flex items-center">
                <span className="text-yellow-500 mr-2">●</span>
                <span>モチベーション維持</span>
              </div>
            </div>
          </div>

          {/* チームワーク */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">チームワーク</h3>
            <div className="space-y-2">
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>協力姿勢</span>
              </div>
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>役割理解</span>
              </div>
              <div className="flex items-center">
                <span className="text-green-500 mr-2">●</span>
                <span>貢献度</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
