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
          <h2 className="text-3xl font-bold text-center mb-8">主な特徴</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="flex items-start space-x-4">
              <div className="flex-1">
                <h3 className="text-xl font-semibold mb-2">強化学習による適応型システム</h3>
                <p className="text-gray-600">
                  学習者の行動パターンを分析し、最適な学習パスを提案します。
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-4">
              <div className="flex-1">
                <h3 className="text-xl font-semibold mb-2">感情分析による学習支援</h3>
                <p className="text-gray-600">
                  学習中の感情状態を分析し、モチベーション維持をサポートします。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* フッター */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <p>&copy; 2024 非認知能力学習プラットフォーム. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </main>
  );
}
