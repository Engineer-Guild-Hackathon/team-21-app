import Image from 'next/legacy/image';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-100 to-white pt-16">
      {/* ヘッダーセクション */}
      <div className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-gray-900">非認知能力学習プラットフォーム</h1>
          <p className="mt-2 text-xl text-gray-600">AIを活用した効果的な学習支援システム</p>
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="container mx-auto px-4 py-12">
        {/* 機能セクション */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">個別最適化学習</h2>
            <p className="text-gray-600">
              AIが一人ひとりの学習進度や特性に合わせて、最適な学習コンテンツを提供します。
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">リアルタイムフィードバック</h2>
            <p className="text-gray-600">
              学習中の行動や反応をAIが分析し、即座にフィードバックを提供します。
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">進捗管理</h2>
            <p className="text-gray-600">詳細な学習記録と分析結果を確認できます。</p>
          </div>
        </div>

        {/* 特徴セクション */}
        <div className="mt-16">
          <h2 className="text-3xl font-bold text-center mb-12">主な特徴</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div className="flex items-center">
              <div className="relative w-24 h-24 flex-shrink-0">
                <Image
                  src="/images/ai-analysis.svg"
                  alt="AI分析"
                  layout="fill"
                  objectFit="contain"
                  priority
                />
              </div>
              <div className="ml-6">
                <h3 className="text-xl font-semibold mb-2">AI感情分析</h3>
                <p className="text-gray-600">
                  学習中の感情状態をリアルタイムで分析し、最適なサポートを提供します。
                </p>
              </div>
            </div>
            <div className="flex items-center">
              <div className="relative w-24 h-24 flex-shrink-0">
                <Image
                  src="/images/adaptive-learning.svg"
                  alt="適応型学習"
                  layout="fill"
                  objectFit="contain"
                />
              </div>
              <div className="ml-6">
                <h3 className="text-xl font-semibold mb-2">適応型学習システム</h3>
                <p className="text-gray-600">
                  学習の進捗に応じて、コンテンツの難易度を自動調整します。
                </p>
              </div>
            </div>
            <div className="flex items-center">
              <div className="relative w-24 h-24 flex-shrink-0">
                <Image
                  src="/images/feedback.svg"
                  alt="フィードバック"
                  layout="fill"
                  objectFit="contain"
                />
              </div>
              <div className="ml-6">
                <h3 className="text-xl font-semibold mb-2">詳細なフィードバック</h3>
                <p className="text-gray-600">学習の成果と改善点を分かりやすく可視化します。</p>
              </div>
            </div>
            <div className="flex items-center">
              <div className="relative w-24 h-24 flex-shrink-0">
                <Image
                  src="/images/progress.svg"
                  alt="進捗管理"
                  layout="fill"
                  objectFit="contain"
                />
              </div>
              <div className="ml-6">
                <h3 className="text-xl font-semibold mb-2">進捗管理ダッシュボード</h3>
                <p className="text-gray-600">
                  学習の進捗状況をグラフや図表で分かりやすく表示します。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
