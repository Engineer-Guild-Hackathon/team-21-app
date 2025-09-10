'use client';

import {
  ChartBarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

export default function FeedbackPage() {
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

  // デモ用のフィードバックデータ
  const feedbackData = {
    recent: [
      {
        id: 1,
        type: 'positive',
        title: 'コミュニケーションスキル',
        content:
          '積極的に意見を述べる姿勢が見られました。さらに相手の意見に耳を傾ける機会を増やすことで、より効果的なコミュニケーションが可能になるでしょう。',
        date: '2024-01-15 15:30',
        points: 85,
      },
      {
        id: 2,
        type: 'improvement',
        title: '感情コントロール',
        content:
          'ストレスを感じた時の対処法について、もう少し練習が必要です。深呼吸や数秒間の休憩を取ることで、気持ちを落ち着かせることができます。',
        date: '2024-01-14 10:15',
        points: 65,
      },
    ],
    skills: [
      {
        name: 'コミュニケーションスキル',
        level: 'good',
        progress: 85,
        feedback: '積極的な発言ができています',
        improvements: ['傾聴スキルの向上', '非言語コミュニケーションの活用'],
      },
      {
        name: '感情コントロール',
        level: 'needs_improvement',
        progress: 65,
        feedback: 'ストレス管理に課題があります',
        improvements: ['深呼吸の練習', '感情の表現方法の学習'],
      },
      {
        name: '目標設定とモチベーション',
        level: 'excellent',
        progress: 92,
        feedback: '目標設定が上手にできています',
        improvements: ['長期目標の設定', '進捗の可視化'],
      },
      {
        name: 'チームワーク',
        level: 'good',
        progress: 78,
        feedback: '協力姿勢が見られます',
        improvements: ['リーダーシップの発揮', '役割分担の理解'],
      },
    ],
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'excellent':
        return 'text-green-600 bg-green-100';
      case 'good':
        return 'text-blue-600 bg-blue-100';
      case 'needs_improvement':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getLevelText = (level: string) => {
    switch (level) {
      case 'excellent':
        return '優秀';
      case 'good':
        return '良好';
      case 'needs_improvement':
        return '要改善';
      default:
        return '不明';
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">学習フィードバック</h1>
          <p className="mt-2 text-sm text-gray-600">
            AIが分析したあなたの学習データに基づいたフィードバックをお届けします
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 最新のフィードバック */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">最新のフィードバック</h2>
          </div>
          <div className="p-6">
            <div className="space-y-6">
              {feedbackData.recent.map(feedback => (
                <div key={feedback.id} className="border-l-4 border-blue-500 pl-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-medium text-gray-900">{feedback.title}</h3>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-500">{feedback.points}点</span>
                      {feedback.type === 'positive' ? (
                        <CheckCircleIcon className="h-5 w-5 text-green-500" />
                      ) : (
                        <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                      )}
                    </div>
                  </div>
                  <p className="text-gray-600 mb-2">{feedback.content}</p>
                  <p className="text-sm text-gray-500">{feedback.date}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* スキル別フィードバック */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {feedbackData.skills.map((skill, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">{skill.name}</h3>
                <span
                  className={`px-2 py-1 text-xs font-semibold rounded-full ${getLevelColor(skill.level)}`}
                >
                  {getLevelText(skill.level)}
                </span>
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">進捗</span>
                  <span className="text-sm font-medium text-gray-900">{skill.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${skill.progress}%` }}
                  ></div>
                </div>
              </div>

              <div className="mb-4">
                <p className="text-sm text-gray-600 mb-2">フィードバック:</p>
                <p className="text-sm text-gray-900">{skill.feedback}</p>
              </div>

              <div>
                <p className="text-sm text-gray-600 mb-2">改善点:</p>
                <ul className="space-y-1">
                  {skill.improvements.map((improvement, idx) => (
                    <li key={idx} className="flex items-center text-sm text-gray-700">
                      <LightBulbIcon className="h-4 w-4 text-yellow-500 mr-2" />
                      {improvement}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        {/* 今週のサマリー */}
        <div className="mt-8 bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">今週のサマリー</h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-3">
                  <StarIcon className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">学習時間</h3>
                <p className="text-2xl font-bold text-green-600">12時間</p>
                <p className="text-sm text-gray-500">前週比 +2時間</p>
              </div>
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-3">
                  <CheckCircleIcon className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">完了レッスン</h3>
                <p className="text-2xl font-bold text-blue-600">8個</p>
                <p className="text-sm text-gray-500">目標: 10個</p>
              </div>
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-purple-100 mb-3">
                  <ChartBarIcon className="h-6 w-6 text-purple-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">平均スコア</h3>
                <p className="text-2xl font-bold text-purple-600">85点</p>
                <p className="text-sm text-gray-500">前週比 +5点</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
