import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'フィードバック - 非認知能力学習プラットフォーム',
  description: 'AIによる学習フィードバック',
};

export default function FeedbackPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">学習フィードバック</h1>

      <div className="w-full max-w-7xl">
        {/* 最新のフィードバック */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">最新のフィードバック</h2>
          <div className="space-y-4">
            <div className="border-l-4 border-green-500 pl-4">
              <p className="text-gray-600">
                コミュニケーションスキルのレッスンでは、積極的に意見を述べる姿勢が見られました。
                さらに相手の意見に耳を傾ける機会を増やすことで、より効果的なコミュニケーションが可能になるでしょう。
              </p>
              <p className="text-sm text-gray-500 mt-2">2024/02/18 15:30</p>
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
