'use client';

import {
  AcademicCapIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  HeartIcon,
  LightBulbIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

export default function AdvicePage() {
  const { user } = useAuth();

  // 保護者以外のアクセスを制限
  if (user?.role !== 'parent') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">アクセス権限がありません</h1>
          <p className="text-gray-600">このページは保護者専用です。</p>
        </div>
      </div>
    );
  }

  // デモ用のアドバイスデータ
  const adviceCategories = [
    {
      id: 'learning',
      title: '学習方法',
      icon: AcademicCapIcon,
      color: 'blue',
      advice: [
        {
          title: '集中力向上のコツ',
          content:
            'お子様は朝の時間帯に最も集中力が高いことが分かっています。重要な学習は午前中にスケジュールすることをお勧めします。',
          priority: 'high',
          date: '2024-01-15',
        },
        {
          title: '復習のタイミング',
          content:
            '学習した内容は24時間以内に復習することで、記憶の定着率が大幅に向上します。毎日の復習時間を確保しましょう。',
          priority: 'medium',
          date: '2024-01-14',
        },
        {
          title: '休憩の取り方',
          content: '45分の学習後には15分の休憩を取ることで、集中力を持続させることができます。',
          priority: 'low',
          date: '2024-01-13',
        },
      ],
    },
    {
      id: 'emotion',
      title: '感情サポート',
      icon: HeartIcon,
      color: 'red',
      advice: [
        {
          title: 'ポジティブな声かけ',
          content:
            'お子様の努力を具体的に褒めることで、自己肯定感が向上します。「頑張っているね」よりも「計算が正確になったね」のように具体的に伝えましょう。',
          priority: 'high',
          date: '2024-01-15',
        },
        {
          title: 'ストレス管理',
          content:
            '学習にストレスを感じている時は、短時間の散歩や音楽鑑賞などの気分転換を提案してください。',
          priority: 'medium',
          date: '2024-01-14',
        },
      ],
    },
    {
      id: 'motivation',
      title: 'やる気向上',
      icon: LightBulbIcon,
      color: 'yellow',
      advice: [
        {
          title: '目標設定のサポート',
          content:
            '小さな目標を設定し、達成した時の喜びを一緒に味わうことで、継続的なやる気を維持できます。',
          priority: 'high',
          date: '2024-01-15',
        },
        {
          title: '興味の拡大',
          content: 'お子様の興味のある分野から学習を始めることで、他の科目への関心も高まります。',
          priority: 'medium',
          date: '2024-01-13',
        },
      ],
    },
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getPriorityText = (priority: string) => {
    switch (priority) {
      case 'high':
        return '高';
      case 'medium':
        return '中';
      case 'low':
        return '低';
      default:
        return '不明';
    }
  };

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'blue':
        return 'text-blue-600 bg-blue-100';
      case 'red':
        return 'text-red-600 bg-red-100';
      case 'yellow':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">アドバイス</h1>
          <p className="mt-2 text-sm text-gray-600">
            AIが分析したお子様の学習データに基づいた、個別のアドバイスをお届けします
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 今日の推奨アクション */}
        <div className="bg-white shadow rounded-lg mb-8">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              今日の推奨アクション
            </h3>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center">
                  <CheckCircleIcon className="h-5 w-5 text-blue-600 mr-2" />
                  <span className="text-sm font-medium text-blue-800">朝の学習時間を確保</span>
                </div>
                <p className="mt-2 text-sm text-blue-700">お子様の集中力が最も高い時間帯です</p>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="flex items-center">
                  <HeartIcon className="h-5 w-5 text-green-600 mr-2" />
                  <span className="text-sm font-medium text-green-800">具体的な褒め言葉を準備</span>
                </div>
                <p className="mt-2 text-sm text-green-700">
                  努力を具体的に認めることでやる気が向上します
                </p>
              </div>
              <div className="p-4 bg-yellow-50 rounded-lg">
                <div className="flex items-center">
                  <ClockIcon className="h-5 w-5 text-yellow-600 mr-2" />
                  <span className="text-sm font-medium text-yellow-800">45分学習+15分休憩</span>
                </div>
                <p className="mt-2 text-sm text-yellow-700">
                  集中力を維持するための理想的なサイクルです
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* カテゴリ別アドバイス */}
        <div className="space-y-8">
          {adviceCategories.map(category => (
            <div key={category.id} className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center mb-4">
                  <div className={`p-2 rounded-lg ${getColorClasses(category.color)}`}>
                    <category.icon className="h-6 w-6" />
                  </div>
                  <h3 className="ml-3 text-lg leading-6 font-medium text-gray-900">
                    {category.title}
                  </h3>
                </div>

                <div className="space-y-4">
                  {category.advice.map((item, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h4 className="text-sm font-medium text-gray-900 mb-2">{item.title}</h4>
                          <p className="text-sm text-gray-600 mb-3">{item.content}</p>
                          <div className="flex items-center text-xs text-gray-500">
                            <ClockIcon className="h-4 w-4 mr-1" />
                            {item.date}
                          </div>
                        </div>
                        <div className="ml-4">
                          <span
                            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getPriorityColor(item.priority)}`}
                          >
                            優先度: {getPriorityText(item.priority)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* 週間サマリー */}
        <div className="mt-8 bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">今週のサマリー</h3>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-3">良い傾向</h4>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-2" />
                    集中力が向上しています
                  </li>
                  <li className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-2" />
                    数学の理解度が高まっています
                  </li>
                  <li className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-2" />
                    学習習慣が定着しています
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-3">改善のポイント</h4>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm text-yellow-600">
                    <ExclamationTriangleIcon className="h-4 w-4 mr-2" />
                    理科の復習時間を増やしましょう
                  </li>
                  <li className="flex items-center text-sm text-yellow-600">
                    <ExclamationTriangleIcon className="h-4 w-4 mr-2" />
                    夜の学習時間を少し早めに設定
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
