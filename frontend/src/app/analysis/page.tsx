'use client';

import {
  AcademicCapIcon,
  ChartBarIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  HeartIcon,
  TrophyIcon,
} from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface ChildProgress {
  id: string;
  name: string;
  grade: string;
  totalLearningTime: number;
  completedQuests: number;
  totalQuests: number;
  emotionalTrend: 'positive' | 'neutral' | 'negative';
  recentActivities: Activity[];
  achievements: Achievement[];
  concerns: Concern[];
}

interface Activity {
  id: string;
  type: 'quest' | 'chat' | 'feedback';
  title: string;
  timestamp: Date;
  duration?: number;
  score?: number;
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  earnedAt: Date;
  category: 'academic' | 'emotional' | 'social';
}

interface Concern {
  id: string;
  type: 'academic' | 'emotional' | 'behavioral';
  description: string;
  severity: 'low' | 'medium' | 'high';
  detectedAt: Date;
}

export default function AnalysisPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [childProgress, setChildProgress] = useState<ChildProgress | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      router.replace('/auth/login?redirect=/analysis');
      return;
    }

    if (user.role !== 'parent' && user.role !== 'teacher') {
      router.replace('/dashboard');
      return;
    }

    // モックデータを読み込み
    loadChildProgress();
  }, [user, router]);

  const loadChildProgress = async () => {
    // 実際の実装ではAPIからデータを取得
    const mockData: ChildProgress = {
      id: '1',
      name: '山田太郎',
      grade: '小学5年生',
      totalLearningTime: 120, // 分
      completedQuests: 15,
      totalQuests: 20,
      emotionalTrend: 'positive',
      recentActivities: [
        {
          id: '1',
          type: 'quest',
          title: '数学：分数の計算',
          timestamp: new Date('2024-01-15T14:30:00'),
          duration: 25,
          score: 85,
        },
        {
          id: '2',
          type: 'chat',
          title: 'AIチューターとの会話',
          timestamp: new Date('2024-01-15T13:45:00'),
          duration: 15,
        },
        {
          id: '3',
          type: 'feedback',
          title: '感情分析レポート',
          timestamp: new Date('2024-01-15T13:30:00'),
        },
      ],
      achievements: [
        {
          id: '1',
          title: '継続学習マスター',
          description: '7日連続で学習を継続',
          earnedAt: new Date('2024-01-14'),
          category: 'academic',
        },
        {
          id: '2',
          title: '感情マネジメント',
          description: '困難な問題でも前向きに取り組む',
          earnedAt: new Date('2024-01-12'),
          category: 'emotional',
        },
      ],
      concerns: [
        {
          id: '1',
          type: 'academic',
          description: '分数の計算でつまずきが見られます',
          severity: 'low',
          detectedAt: new Date('2024-01-15'),
        },
      ],
    };

    setChildProgress(mockData);
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">データを読み込み中...</p>
        </div>
      </div>
    );
  }

  if (!childProgress) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
          <p className="text-gray-600">データが見つかりません</p>
        </div>
      </div>
    );
  }

  const progressPercentage = (childProgress.completedQuests / childProgress.totalQuests) * 100;

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">
                {user?.role === 'parent' ? '子どもの学習分析' : '生徒分析'}
              </h1>
              <p className="mt-2 text-gray-600">
                {childProgress.name} ({childProgress.grade})
              </p>
            </div>
            <div className="flex space-x-4">
              <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                詳細レポート
              </button>
              <button className="bg-gray-200 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-300">
                エクスポート
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 統計カード */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <ClockIcon className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">総学習時間</p>
                <p className="text-2xl font-bold text-gray-900">
                  {childProgress.totalLearningTime}分
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <AcademicCapIcon className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">完了クエスト</p>
                <p className="text-2xl font-bold text-gray-900">
                  {childProgress.completedQuests}/{childProgress.totalQuests}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <HeartIcon className="h-8 w-8 text-pink-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">感情状態</p>
                <p className="text-2xl font-bold text-gray-900">
                  {childProgress.emotionalTrend === 'positive' && '😊 良好'}
                  {childProgress.emotionalTrend === 'neutral' && '😐 普通'}
                  {childProgress.emotionalTrend === 'negative' && '😔 要注意'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <TrophyIcon className="h-8 w-8 text-yellow-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">獲得実績</p>
                <p className="text-2xl font-bold text-gray-900">
                  {childProgress.achievements.length}個
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 進捗チャート */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">学習進捗</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>クエスト完了率</span>
                  <span>{Math.round(progressPercentage)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progressPercentage}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* 最近の活動 */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">最近の活動</h2>
            <div className="space-y-3">
              {childProgress.recentActivities.map(activity => (
                <div
                  key={activity.id}
                  className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex-shrink-0">
                    {activity.type === 'quest' && (
                      <AcademicCapIcon className="h-5 w-5 text-blue-600" />
                    )}
                    {activity.type === 'chat' && (
                      <ChartBarIcon className="h-5 w-5 text-green-600" />
                    )}
                    {activity.type === 'feedback' && (
                      <HeartIcon className="h-5 w-5 text-pink-600" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">{activity.title}</p>
                    <p className="text-sm text-gray-500">
                      {activity.timestamp.toLocaleDateString('ja-JP')}{' '}
                      {activity.timestamp.toLocaleTimeString('ja-JP', {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                      {activity.duration && ` • ${activity.duration}分`}
                      {activity.score && ` • スコア: ${activity.score}`}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 実績 */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">獲得実績</h2>
            <div className="space-y-3">
              {childProgress.achievements.map(achievement => (
                <div
                  key={achievement.id}
                  className="flex items-center space-x-3 p-3 bg-yellow-50 rounded-lg"
                >
                  <TrophyIcon className="h-5 w-5 text-yellow-600 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">{achievement.title}</p>
                    <p className="text-sm text-gray-500">{achievement.description}</p>
                    <p className="text-xs text-gray-400">
                      {achievement.earnedAt.toLocaleDateString('ja-JP')} 獲得
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 注意事項 */}
          {childProgress.concerns.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">注意事項</h2>
              <div className="space-y-3">
                {childProgress.concerns.map(concern => (
                  <div
                    key={concern.id}
                    className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg"
                  >
                    <ExclamationTriangleIcon className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900">{concern.description}</p>
                      <p className="text-xs text-gray-500">
                        {concern.detectedAt.toLocaleDateString('ja-JP')} 検出
                      </p>
                    </div>
                    <span
                      className={`px-2 py-1 text-xs rounded-full ${
                        concern.severity === 'high'
                          ? 'bg-red-100 text-red-800'
                          : concern.severity === 'medium'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {concern.severity === 'high'
                        ? '高'
                        : concern.severity === 'medium'
                          ? '中'
                          : '低'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
