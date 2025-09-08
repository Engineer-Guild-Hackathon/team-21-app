'use client';

import EmotionAnalysis from '../components/EmotionAnalysis';

export default function LearningPage() {
  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">学習ページ</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">感情分析</h2>
          <EmotionAnalysis />
        </div>

        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">学習コンテンツ</h2>
          <div className="bg-white rounded-lg shadow p-6">
            <p className="text-gray-600">
              感情分析の結果に基づいて、最適な学習コンテンツが表示されます。
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
