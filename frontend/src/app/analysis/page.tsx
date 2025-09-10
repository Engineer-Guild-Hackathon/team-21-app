'use client';

import {
  AcademicCapIcon,
  ArrowTrendingUpIcon,
  ClockIcon,
  HeartIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

export default function AnalysisPage() {
  const { user } = useAuth();

  // 保護者と教師のみアクセス可能
  if (user?.role !== 'parent' && user?.role !== 'teacher') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">アクセス権限がありません</h1>
          <p className="text-gray-600">このページは保護者・教師専用です。</p>
        </div>
      </div>
    );
  }

  const isParent = user?.role === 'parent';
  const pageTitle = isParent ? '行動分析' : '生徒分析';
  const pageDescription = isParent
    ? 'お子様の学習行動と感情の変化を分析します'
    : 'クラス全体の学習状況と個別の成長を分析します';

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">{pageTitle}</h1>
          <p className="mt-2 text-sm text-gray-600">{pageDescription}</p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 主要指標 */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 mb-8">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <HeartIcon className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {isParent ? '感情スコア' : '平均感情スコア'}
                    </dt>
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
                  <AcademicCapIcon className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {isParent ? '学習時間' : '総学習時間'}
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {isParent ? '12時間' : '156時間'}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ArrowTrendingUpIcon className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {isParent ? '成長率' : '平均成長率'}
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">+12%</dd>
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
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {isParent ? '集中時間' : '平均集中時間'}
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {isParent ? '45分' : '38分'}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* メイン分析エリア */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 mb-8">
          {/* 感情分析グラフ */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">感情分析</h3>
              <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
                <p className="text-gray-500">感情分析グラフがここに表示されます</p>
              </div>
            </div>
          </div>

          {/* 学習パターン分析 */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">学習パターン</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">朝の学習</span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '75%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">75%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">午後の学習</span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '60%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">60%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">夜の学習</span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '45%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900">45%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 詳細分析 */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* 科目別分析 */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">科目別分析</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">数学</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-500 h-2 rounded-full" style={{ width: '90%' }}></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">90%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">国語</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">85%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">理科</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-yellow-500 h-2 rounded-full"
                        style={{ width: '78%' }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">78%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">社会</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-500 h-2 rounded-full"
                        style={{ width: '82%' }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">82%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 非認知能力分析 */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">非認知能力</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">やり抜く力</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div className="bg-red-500 h-2 rounded-full" style={{ width: '88%' }}></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">88%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">協調性</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-500 h-2 rounded-full" style={{ width: '92%' }}></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">92%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">好奇心</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">85%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">自己管理</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-yellow-500 h-2 rounded-full"
                        style={{ width: '80%' }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">80%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 推奨アクション */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">推奨アクション</h3>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    {isParent
                      ? 'お子様の協調性が非常に高いです。グループ学習を増やすことをお勧めします。'
                      : 'クラス全体の協調性が向上しています。チームワークを活かした活動を増やしましょう。'}
                  </p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <p className="text-sm text-green-800">
                    {isParent
                      ? '数学の理解度が高いです。より高度な問題に挑戦させてみてください。'
                      : '数学の平均スコアが高いです。応用問題の導入を検討しましょう。'}
                  </p>
                </div>
                <div className="p-3 bg-yellow-50 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    {isParent
                      ? '自己管理能力の向上に焦点を当てた学習を提案します。'
                      : '一部の生徒の自己管理能力向上が必要です。個別サポートを検討しましょう。'}
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
