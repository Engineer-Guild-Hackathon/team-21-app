import { apiUrl } from '@/lib/api';
('use client');

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface UserStats {
  grit_level: number;
  collaboration_level: number;
  self_regulation_level: number;
  emotional_intelligence_level: number;
}

export default function ProgressPage() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/auth/login?redirect=/progress');
      return;
    }

    fetchUserStats();

    // 5秒ごとに自動更新（最新のユーザー統計を取得）
    const interval = setInterval(() => {
      console.log('🔄 自動更新: ユーザー統計を再取得');
      fetchUserStats();
    }, 5000);

    return () => clearInterval(interval);
  }, [isAuthenticated, router]);

  const fetchUserStats = async () => {
    try {
      console.log('🔍 ユーザー統計取得開始');

      const token = localStorage.getItem('token');
      if (!token) {
        console.error('❌ 認証トークンが見つかりません');
        return;
      }

      console.log('📤 ユーザー統計取得リクエスト送信');

      const response = await fetch('${apiUrl("")}/api/avatars/stats', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      console.log('📥 ユーザー統計レスポンス:', response.status, response.statusText);

      if (response.ok) {
        const stats = await response.json();
        console.log('✅ ユーザー統計取得成功:', stats);
        setUserStats(stats);
      } else {
        const errorText = await response.text();
        console.error('❌ ユーザー統計取得失敗:', response.status, errorText);
      }
    } catch (error) {
      console.error('❌ ユーザー統計取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const calculateOverallProgress = () => {
    if (!userStats) return 0;

    const skills = [
      userStats.grit_level,
      userStats.collaboration_level,
      userStats.self_regulation_level,
      userStats.emotional_intelligence_level,
    ];

    const average = skills.reduce((sum, skill) => sum + skill, 0) / skills.length;
    return Math.round((average / 5.0) * 100); // 5点満点を100%に変換
  };

  const isNewUser = () => {
    if (!userStats) return false;

    // 全てのスキルが初期値（1.0）の場合、新規ユーザーと判定
    return (
      userStats.grit_level === 1.0 &&
      userStats.collaboration_level === 1.0 &&
      userStats.self_regulation_level === 1.0 &&
      userStats.emotional_intelligence_level === 1.0
    );
  };

  const getSkillColor = (level: number) => {
    if (level >= 4) return 'bg-green-500';
    if (level >= 3) return 'bg-yellow-500';
    if (level >= 2) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getSkillPercentage = (level: number) => {
    return Math.round((level / 5.0) * 100);
  };

  if (isLoading) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center">
        <div className="text-xl">進捗を読み込み中...</div>
      </main>
    );
  }

  const overallProgress = calculateOverallProgress();

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">学習進捗</h1>

      {isNewUser() && (
        <div className="w-full max-w-7xl mb-8">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <div className="flex items-center">
              <div className="text-4xl mr-4">🎯</div>
              <div>
                <h2 className="text-xl font-semibold text-blue-900 mb-2">学習を始めましょう！</h2>
                <p className="text-blue-700">
                  AIチャットで質問したり、クエストに挑戦したりして、スキルを向上させていきましょう。
                  学習活動が増えると、ここに進捗が表示されます。
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="w-full max-w-7xl">
        {/* 全体の進捗 */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">全体の進捗</h2>
          <div className="h-4 bg-gray-200 rounded-full">
            <div
              className={`h-4 rounded-full ${getSkillColor(overallProgress / 20)}`}
              style={{ width: `${overallProgress}%` }}
            ></div>
          </div>
          <div className="mt-2 text-gray-600">{overallProgress}% 完了</div>
          <div className="mt-2 text-sm text-gray-500">
            平均スキルレベル:{' '}
            {userStats
              ? (
                  (userStats.grit_level +
                    userStats.collaboration_level +
                    userStats.self_regulation_level +
                    userStats.emotional_intelligence_level) /
                  4
                ).toFixed(1)
              : 0}
            /5.0
          </div>
        </div>

        {/* スキル別の進捗 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* グリット（やり抜く力） */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">グリット（やり抜く力）</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div
                className={`h-4 rounded-full ${getSkillColor(userStats?.grit_level || 0)}`}
                style={{ width: `${getSkillPercentage(userStats?.grit_level || 0)}%` }}
              ></div>
            </div>
            <div className="text-gray-600">
              {getSkillPercentage(userStats?.grit_level || 0)}% 完了
            </div>
            <div className="text-sm text-gray-500 mt-1">
              レベル: {userStats?.grit_level?.toFixed(1) || 0}/5.0
            </div>
          </div>

          {/* 協調性 */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">協調性</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div
                className={`h-4 rounded-full ${getSkillColor(userStats?.collaboration_level || 0)}`}
                style={{ width: `${getSkillPercentage(userStats?.collaboration_level || 0)}%` }}
              ></div>
            </div>
            <div className="text-gray-600">
              {getSkillPercentage(userStats?.collaboration_level || 0)}% 完了
            </div>
            <div className="text-sm text-gray-500 mt-1">
              レベル: {userStats?.collaboration_level?.toFixed(1) || 0}/5.0
            </div>
          </div>

          {/* 自己制御 */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">自己制御</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div
                className={`h-4 rounded-full ${getSkillColor(userStats?.self_regulation_level || 0)}`}
                style={{ width: `${getSkillPercentage(userStats?.self_regulation_level || 0)}%` }}
              ></div>
            </div>
            <div className="text-gray-600">
              {getSkillPercentage(userStats?.self_regulation_level || 0)}% 完了
            </div>
            <div className="text-sm text-gray-500 mt-1">
              レベル: {userStats?.self_regulation_level?.toFixed(1) || 0}/5.0
            </div>
          </div>

          {/* 感情知能 */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">感情知能</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div
                className={`h-4 rounded-full ${getSkillColor(userStats?.emotional_intelligence_level || 0)}`}
                style={{
                  width: `${getSkillPercentage(userStats?.emotional_intelligence_level || 0)}%`,
                }}
              ></div>
            </div>
            <div className="text-gray-600">
              {getSkillPercentage(userStats?.emotional_intelligence_level || 0)}% 完了
            </div>
            <div className="text-sm text-gray-500 mt-1">
              レベル: {userStats?.emotional_intelligence_level?.toFixed(1) || 0}/5.0
            </div>
          </div>

          {/* 推奨アクション */}
          <div className="bg-blue-50 rounded-lg shadow-lg p-6 md:col-span-2">
            <h3 className="text-xl font-semibold mb-4 text-blue-800">AI推奨アクション</h3>
            <div className="space-y-2">
              <p className="text-blue-700">
                {userStats ? (
                  <>
                    {userStats.grit_level < 3 &&
                      '🎯 目標設定クエストに挑戦してやり抜く力を鍛えましょう'}
                    {userStats.collaboration_level < 3 &&
                      '🤝 AIチャットで積極的に質問して協調性を高めましょう'}
                    {userStats.self_regulation_level < 3 &&
                      '⏰ 学習時間を決めて計画的に取り組みましょう'}
                    {userStats.emotional_intelligence_level < 3 &&
                      '💭 自分の感情を振り返る時間を作りましょう'}
                    {userStats.grit_level >= 3 &&
                      userStats.collaboration_level >= 3 &&
                      userStats.self_regulation_level >= 3 &&
                      userStats.emotional_intelligence_level >= 3 &&
                      '🌟 素晴らしいバランスです！新しいクエストに挑戦してみましょう'}
                  </>
                ) : (
                  'データを蓄積中です。AIチャットやクエストを利用してください。'
                )}
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
