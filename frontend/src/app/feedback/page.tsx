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
      console.log('🔍 フィードバック取得開始');

      const token = localStorage.getItem('token');
      if (!token) {
        console.error('❌ 認証トークンが見つかりません');
        return;
      }

      console.log('📤 ML分析結果取得リクエスト送信');

      // 最新のML分析結果を取得
      const response = await fetch('http://localhost:8000/api/ml/latest-analysis', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      console.log('📥 ML分析結果レスポンス:', response.status, response.statusText);

      if (response.ok) {
        const analysisResult = await response.json();
        console.log('✅ ML分析結果取得成功:', analysisResult);

        const mlFeedback: MLFeedback = {
          user_id: analysisResult.user_id,
          skills: analysisResult.skills,
          feedback: analysisResult.feedback,
          analysis_timestamp: analysisResult.analysis_timestamp,
        };

        console.log('📊 フィードバック設定:', mlFeedback);
        setMlFeedback(mlFeedback);
      } else {
        const errorText = await response.text();
        console.error('❌ ML分析結果の取得に失敗しました:', response.status, errorText);
      }
    } catch (error) {
      console.error('❌ フィードバック取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
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
