'use client';

import {
  ArrowTrendingUpIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ClockIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

export default function ProgressPage() {
  const { user } = useAuth();

  // 生徒以外のアクセスを制限
  if (user?.role !== 'student') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">アクセス権限がありません</h1>
          <p className="text-gray-600">このページは生徒専用です。</p>
        </div>
      </div>
    );
  }

  // デモ用の進捗データ
  const progressData = {
    overall: {
      percentage: 75,
      completed: 15,
      total: 20,
      streak: 7,
    },
    skills: [
      {
        name: 'コミュニケーションスキル',
        progress: 85,
        completed: 8,
        total: 10,
        color: 'bg-green-500',
        lastActivity: '2時間前',
      },
      {
        name: '感情コントロール',
        progress: 65,
        completed: 6,
        total: 10,
        color: 'bg-yellow-500',
        lastActivity: '1日前',
      },
      {
        name: '目標設定とモチベーション',
        progress: 92,
        completed: 9,
        total: 10,
        color: 'bg-blue-500',
        lastActivity: '30分前',
      },
      {
        name: 'チームワーク',
        progress: 78,
        completed: 7,
        total: 9,
        color: 'bg-purple-500',
        lastActivity: '3時間前',
      },
      {
        name: '問題解決能力',
        progress: 55,
        completed: 5,
        total: 10,
        color: 'bg-indigo-500',
        lastActivity: '2日前',
      },
      {
        name: 'レジリエンス',
        progress: 40,
        completed: 4,
        total: 10,
        color: 'bg-pink-500',
        lastActivity: '1週間前',
      },
    ],
    weekly: [
      { day: '月', hours: 2.5, completed: 3 },
      { day: '火', hours: 1.8, completed: 2 },
      { day: '水', hours: 3.2, completed: 4 },
      { day: '木', hours: 2.1, completed: 2 },
      { day: '金', hours: 2.8, completed: 3 },
      { day: '土', hours: 1.5, completed: 1 },
      { day: '日', hours: 0, completed: 0 },
    ],
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">学習進捗</h1>
          <p className="mt-2 text-sm text-gray-600">あなたの学習の進捗状況と成長を確認できます</p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 全体の進捗 */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">全体の進捗</h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-3">
                  <ChartBarIcon className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">全体進捗</h3>
                <p className="text-3xl font-bold text-blue-600">
                  {progressData.overall.percentage}%
                </p>
                <p className="text-sm text-gray-500">
                  {progressData.overall.completed}/{progressData.overall.total} レッスン完了
                </p>
              </div>
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-3">
                  <CheckCircleIcon className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">完了レッスン</h3>
                <p className="text-3xl font-bold text-green-600">
                  {progressData.overall.completed}
                </p>
                <p className="text-sm text-gray-500">今週 +3</p>
              </div>
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-yellow-100 mb-3">
                  <StarIcon className="h-6 w-6 text-yellow-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">連続学習日数</h3>
                <p className="text-3xl font-bold text-yellow-600">{progressData.overall.streak}</p>
                <p className="text-sm text-gray-500">日連続</p>
              </div>
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-purple-100 mb-3">
                  <ArrowTrendingUpIcon className="h-6 w-6 text-purple-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">成長率</h3>
                <p className="text-3xl font-bold text-purple-600">+12%</p>
                <p className="text-sm text-gray-500">先月比</p>
              </div>
            </div>
          </div>
        </div>

        {/* スキル別の進捗 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {progressData.skills.map((skill, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">{skill.name}</h3>
                <span className="text-sm text-gray-500">{skill.lastActivity}</span>
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">進捗</span>
                  <span className="text-sm font-medium text-gray-900">{skill.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-300 ${skill.color}`}
                    style={{ width: `${skill.progress}%` }}
                  ></div>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>
                  {skill.completed}/{skill.total} レッスン完了
                </span>
                <div className="flex items-center">
                  <ClockIcon className="h-4 w-4 mr-1" />
                  {skill.lastActivity}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* 週間学習記録 */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">今週の学習記録</h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-7 gap-4">
              {progressData.weekly.map((day, index) => (
                <div key={index} className="text-center">
                  <div className="text-sm font-medium text-gray-900 mb-2">{day.day}</div>
                  <div className="h-20 bg-gray-100 rounded-lg flex flex-col items-center justify-center mb-2">
                    <div className="text-lg font-bold text-gray-900">{day.hours}h</div>
                    <div className="text-xs text-gray-500">{day.completed}完了</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-600">
                今週の総学習時間: <span className="font-medium text-gray-900">14.9時間</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
