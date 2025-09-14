'use client';

export default function ProgressPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">学習進捗</h1>

      <div className="w-full max-w-7xl">
        {/* 全体の進捗 */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">全体の進捗</h2>
          <div className="h-4 bg-gray-200 rounded-full">
            <div className="h-4 bg-blue-500 rounded-full" style={{ width: '60%' }}></div>
          </div>
          <div className="mt-2 text-gray-600">60% 完了</div>
        </div>

        {/* スキル別の進捗 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* コミュニケーションスキル */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">コミュニケーションスキル</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div className="h-4 bg-green-500 rounded-full" style={{ width: '75%' }}></div>
            </div>
            <div className="text-gray-600">75% 完了</div>
          </div>

          {/* 感情コントロール */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">感情コントロール</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div className="h-4 bg-yellow-500 rounded-full" style={{ width: '45%' }}></div>
            </div>
            <div className="text-gray-600">45% 完了</div>
          </div>

          {/* 目標設定とモチベーション */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">目標設定とモチベーション</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div className="h-4 bg-purple-500 rounded-full" style={{ width: '60%' }}></div>
            </div>
            <div className="text-gray-600">60% 完了</div>
          </div>

          {/* チームワーク */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">チームワーク</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div className="h-4 bg-red-500 rounded-full" style={{ width: '80%' }}></div>
            </div>
            <div className="text-gray-600">80% 完了</div>
          </div>

          {/* 問題解決能力 */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">問題解決能力</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div className="h-4 bg-indigo-500 rounded-full" style={{ width: '55%' }}></div>
            </div>
            <div className="text-gray-600">55% 完了</div>
          </div>

          {/* レジリエンス */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">レジリエンス</h3>
            <div className="h-4 bg-gray-200 rounded-full mb-2">
              <div className="h-4 bg-pink-500 rounded-full" style={{ width: '40%' }}></div>
            </div>
            <div className="text-gray-600">40% 完了</div>
          </div>
        </div>
      </div>
    </main>
  );
}
