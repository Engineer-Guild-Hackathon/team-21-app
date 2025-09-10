'use client';

import { AcademicCapIcon, ChartBarIcon, ClockIcon, HeartIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

export default function DashboardPage() {
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

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">保護者ダッシュボード</h1>
          <p className="mt-2 text-sm text-gray-600">お子様の学習進捗と成長を確認できます</p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 統計カード */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 mb-8">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <AcademicCapIcon className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">今週の学習時間</dt>
                    <dd className="text-lg font-medium text-gray-900">12時間</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ChartBarIcon className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">完了した課題</dt>
                    <dd className="text-lg font-medium text-gray-900">24個</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <HeartIcon className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">感情スコア</dt>
                    <dd className="text-lg font-medium text-gray-900">85%</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ClockIcon className="h-6 w-6 text-purple-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">連続学習日数</dt>
                    <dd className="text-lg font-medium text-gray-900">7日</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* メインコンテンツ */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* 学習進捗グラフ */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">学習進捗</h3>
              <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
                <p className="text-gray-500">進捗グラフがここに表示されます</p>
              </div>
            </div>
          </div>

          {/* 感情分析結果 */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">感情分析結果</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">ポジティブ</span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">85%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">集中度</span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '78%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">78%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">やる気</span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '92%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">92%</span>
                </div>
              </div>
            </div>
          </div>

          {/* 最近の活動 */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">最近の活動</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">数学の課題を完了しました</p>
                    <p className="text-xs text-gray-500">2時間前</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">AIチャットで質問しました</p>
                    <p className="text-xs text-gray-500">4時間前</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="h-2 w-2 bg-yellow-500 rounded-full"></div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">新しいクエストを開始しました</p>
                    <p className="text-xs text-gray-500">1日前</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* アドバイス */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                お子様へのアドバイス
              </h3>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    お子様は今週、とても集中して学習に取り組んでいます。この調子で続けていきましょう。
                  </p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <p className="text-sm text-green-800">
                    数学の理解度が向上しています。複雑な問題にも挑戦してみることをお勧めします。
                  </p>
                </div>
                <div className="p-3 bg-yellow-50 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    休憩時間を適切に取ることで、より効率的な学習ができるでしょう。
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
