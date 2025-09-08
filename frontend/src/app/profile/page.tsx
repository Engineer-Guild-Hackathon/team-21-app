'use client';

import {
  AcademicCapIcon,
  ChartBarIcon,
  HeartIcon,
  LightBulbIcon,
  StarIcon,
  UserGroupIcon,
} from '@heroicons/react/24/outline';
import { useState } from 'react';

interface NonCogSkill {
  name: string;
  score: number;
  icon: any;
  color: string;
  description: string;
  recentActivity: string;
}

export default function ProfilePage() {
  const [activeTab, setActiveTab] = useState('overview');

  const nonCogSkills: NonCogSkill[] = [
    {
      name: 'やり抜く力',
      score: 0.85,
      icon: StarIcon,
      color: 'bg-yellow-100 text-yellow-800',
      description: '難しい課題に粘り強く取り組む力が育っています。',
      recentActivity: '「植物図鑑ミッション」で、失敗を重ねながらも最後まで取り組みました。',
    },
    {
      name: '協調性',
      score: 0.75,
      icon: UserGroupIcon,
      color: 'bg-blue-100 text-blue-800',
      description: 'チームでの活動に積極的に参加しています。',
      recentActivity: '「宝探しクエスト」でチームメンバーと協力して課題を解決しました。',
    },
    {
      name: '好奇心',
      score: 0.9,
      icon: LightBulbIcon,
      color: 'bg-purple-100 text-purple-800',
      description: '新しいことへの興味関心が高まっています。',
      recentActivity: 'AIキャラクターに多くの質問をして、理解を深めようとしています。',
    },
    {
      name: '感情制御',
      score: 0.7,
      icon: HeartIcon,
      color: 'bg-red-100 text-red-800',
      description: '感情をコントロールする力が育ってきています。',
      recentActivity: 'フラストレーションを感じた時も、落ち着いて対処できました。',
    },
    {
      name: '学習意欲',
      score: 0.8,
      icon: AcademicCapIcon,
      color: 'bg-green-100 text-green-800',
      description: '自主的に学習に取り組む姿勢が見られます。',
      recentActivity: '毎日継続的にクエストに挑戦しています。',
    },
    {
      name: '分析力',
      score: 0.65,
      icon: ChartBarIcon,
      color: 'bg-indigo-100 text-indigo-800',
      description: '問題を論理的に考える力が育っています。',
      recentActivity: '複雑なパズルを段階的に解決することができました。',
    },
  ];

  const recentAchievements = [
    {
      title: '植物図鑑マスター',
      date: '2024/03/10',
      description: '50種類の植物を正しく分類しました',
      type: 'achievement',
    },
    {
      title: 'チームリーダー',
      date: '2024/03/08',
      description: 'グループクエストで3回リーダーを務めました',
      type: 'leadership',
    },
    {
      title: '継続の達人',
      date: '2024/03/05',
      description: '30日連続でログインを達成しました',
      type: 'streak',
    },
  ];

  const learningHistory = [
    {
      date: '2024/03/10',
      activity: '植物図鑑ミッション',
      duration: '45分',
      emotions: ['集中', '喜び'],
      achievement: '全問正解',
    },
    {
      date: '2024/03/09',
      activity: 'チーム宝探しクエスト',
      duration: '30分',
      emotions: ['協力', '興奮'],
      achievement: 'チーム1位',
    },
    {
      date: '2024/03/08',
      activity: '算数パズル',
      duration: '20分',
      emotions: ['フラストレーション', '達成感'],
      achievement: 'クリア',
    },
  ];

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* プロフィールヘッダー */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex items-center space-x-6">
            <div className="h-24 w-24 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 flex items-center justify-center">
              <span className="text-3xl font-bold text-white">YK</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">山田 健太</h1>
              <p className="text-gray-500">小学5年生</p>
              <div className="mt-2 flex items-center space-x-4">
                <div className="flex items-center">
                  <StarIcon className="h-5 w-5 text-yellow-400 mr-1" />
                  <span className="text-sm text-gray-600">レベル 15</span>
                </div>
                <div className="flex items-center">
                  <LightBulbIcon className="h-5 w-5 text-purple-400 mr-1" />
                  <span className="text-sm text-gray-600">クエスト達成率 85%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* タブナビゲーション */}
        <div className="border-b border-gray-200 mb-8">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('overview')}
              className={`${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              概要
            </button>
            <button
              onClick={() => setActiveTab('achievements')}
              className={`${
                activeTab === 'achievements'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              達成状況
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`${
                activeTab === 'history'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              学習履歴
            </button>
          </nav>
        </div>

        {/* コンテンツエリア */}
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* 非認知能力スコア */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {nonCogSkills.map(skill => (
                <div key={skill.name} className={`rounded-lg p-6 ${skill.color}`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center">
                      <skill.icon className="h-6 w-6 mr-2" />
                      <h3 className="font-semibold">{skill.name}</h3>
                    </div>
                    <div className="text-2xl font-bold">{Math.round(skill.score * 100)}</div>
                  </div>
                  <div className="mb-4">
                    <div className="w-full bg-white rounded-full h-2">
                      <div
                        className="bg-current h-2 rounded-full transition-all duration-500"
                        style={{ width: `${skill.score * 100}%` }}
                      />
                    </div>
                  </div>
                  <p className="text-sm mb-2">{skill.description}</p>
                  <p className="text-sm opacity-75">{skill.recentActivity}</p>
                </div>
              ))}
            </div>

            {/* 最近の達成 */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">最近の達成</h2>
              <div className="space-y-4">
                {recentAchievements.map(achievement => (
                  <div
                    key={achievement.title}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <h3 className="font-medium text-gray-900">{achievement.title}</h3>
                      <p className="text-sm text-gray-500">{achievement.description}</p>
                    </div>
                    <div className="text-sm text-gray-500">{achievement.date}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'achievements' && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">獲得バッジ一覧</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="w-16 h-16 mx-auto mb-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 flex items-center justify-center">
                    <StarIcon className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="font-medium text-gray-900">バッジ {i + 1}</h3>
                  <p className="text-sm text-gray-500">達成条件の説明</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">学習履歴</h2>
            <div className="space-y-4">
              {learningHistory.map((item, index) => (
                <div key={index} className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-900">{item.activity}</h3>
                    <span className="text-sm text-gray-500">{item.date}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <div>学習時間: {item.duration}</div>
                    <div>感情: {item.emotions.join(', ')}</div>
                    <div>結果: {item.achievement}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
