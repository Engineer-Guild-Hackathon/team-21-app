import { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "学習 - 非認知能力学習プラットフォーム",
  description: "非認知能力を育む学習コンテンツ",
};

export default function LearningPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-4xl font-bold mb-8">学習コンテンツ</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-7xl">
        {/* コミュニケーションスキル */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">
            コミュニケーションスキル
          </h2>
          <p className="text-gray-600 mb-4">
            効果的なコミュニケーション方法を学び、対人関係を円滑にします。
          </p>
          <Link
            href="/learning/communication"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
          >
            レッスンを始める
          </Link>
        </div>

        {/* 感情コントロール */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">感情コントロール</h2>
          <p className="text-gray-600 mb-4">
            感情を理解し、適切にコントロールする方法を学びます。
          </p>
          <Link
            href="/learning/emotion-control"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
          >
            レッスンを始める
          </Link>
        </div>

        {/* 目標設定とモチベーション */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">
            目標設定とモチベーション
          </h2>
          <p className="text-gray-600 mb-4">
            効果的な目標設定とモチベーション維持の方法を学びます。
          </p>
          <Link
            href="/learning/goal-setting"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
          >
            レッスンを始める
          </Link>
        </div>

        {/* チームワーク */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">チームワーク</h2>
          <p className="text-gray-600 mb-4">
            効果的なチーム協力とリーダーシップスキルを身につけます。
          </p>
          <Link
            href="/learning/teamwork"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
          >
            レッスンを始める
          </Link>
        </div>

        {/* 問題解決能力 */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">問題解決能力</h2>
          <p className="text-gray-600 mb-4">
            創造的な問題解決アプローチと批判的思考を学びます。
          </p>
          <Link
            href="/learning/problem-solving"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
          >
            レッスンを始める
          </Link>
        </div>

        {/* レジリエンス */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">レジリエンス</h2>
          <p className="text-gray-600 mb-4">
            困難に立ち向かい、回復する力を身につけます。
          </p>
          <Link
            href="/learning/resilience"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
          >
            レッスンを始める
          </Link>
        </div>
      </div>

      {/* 学習ガイド */}
      <div className="mt-12 w-full max-w-7xl">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-semibold mb-4">学習の進め方</h2>
          <div className="space-y-4">
            <div className="flex items-start">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                1
              </div>
              <div className="ml-4">
                <h3 className="font-medium">興味のあるスキルを選択</h3>
                <p className="text-gray-600">
                  上記の6つのスキルから、最も興味のあるものや必要性を感じるものを選んでください。
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                2
              </div>
              <div className="ml-4">
                <h3 className="font-medium">レッスンに取り組む</h3>
                <p className="text-gray-600">
                  各レッスンは理論学習、実践演習、振り返りの3つのステップで構成されています。
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                3
              </div>
              <div className="ml-4">
                <h3 className="font-medium">フィードバックを確認</h3>
                <p className="text-gray-600">
                  AIによる分析結果とフィードバックを確認し、次のステップに活かしましょう。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
