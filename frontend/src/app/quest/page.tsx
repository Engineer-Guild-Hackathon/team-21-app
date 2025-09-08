'use client';

import { CheckCircleIcon, ClockIcon, StarIcon } from '@heroicons/react/24/outline';
import { useState } from 'react';

export default function QuestPage() {
  const [currentStep, setCurrentStep] = useState(1);

  const steps = [
    {
      id: 1,
      title: '問題を理解する',
      description: '与えられた課題の目的と要件を理解しましょう。',
      completed: currentStep > 1,
    },
    {
      id: 2,
      title: '解決策を考える',
      description: '複数の解決アプローチを検討し、最適な方法を選びましょう。',
      completed: currentStep > 2,
    },
    {
      id: 3,
      title: '実行する',
      description: '選んだ解決策を実行に移しましょう。',
      completed: currentStep > 3,
    },
    {
      id: 4,
      title: '振り返る',
      description: '結果を評価し、学んだことを整理しましょう。',
      completed: currentStep > 4,
    },
  ];

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* クエストヘッダー */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">やり抜く力を育むクエスト</h1>
              <p className="mt-2 text-gray-600">難しい課題に挑戦して、諦めない心を育てましょう。</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <ClockIcon className="h-5 w-5 text-gray-400 mr-1" />
                <span className="text-sm text-gray-600">推定時間: 30分</span>
              </div>
              <div className="flex items-center">
                <StarIcon className="h-5 w-5 text-yellow-400 mr-1" />
                <span className="text-sm text-gray-600">難易度: 中級</span>
              </div>
            </div>
          </div>
        </div>

        {/* ステップナビゲーション */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex-1">
                <div className="relative">
                  <div
                    className={`
                    h-12 w-12 rounded-full flex items-center justify-center
                    ${step.completed ? 'bg-green-100' : currentStep === step.id ? 'bg-blue-100' : 'bg-gray-100'}
                    mx-auto
                  `}
                  >
                    {step.completed ? (
                      <CheckCircleIcon className="h-6 w-6 text-green-600" />
                    ) : (
                      <span
                        className={`text-lg font-semibold ${currentStep === step.id ? 'text-blue-600' : 'text-gray-500'}`}
                      >
                        {step.id}
                      </span>
                    )}
                  </div>
                  <div className="mt-2 text-center">
                    <div className="text-sm font-medium text-gray-900">{step.title}</div>
                    <div className="text-xs text-gray-500 mt-1">{step.description}</div>
                  </div>
                  {index < steps.length - 1 && (
                    <div className="absolute top-6 left-1/2 w-full h-0.5 bg-gray-200" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* クエストコンテンツ */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="prose max-w-none">
            <h2 className="text-xl font-semibold mb-4">
              現在のステップ: {steps[currentStep - 1].title}
            </h2>

            {currentStep === 1 && (
              <div className="space-y-4">
                <p>
                  このクエストでは、難しい課題に直面したときの対処方法を学びます。
                  以下の状況を想定してみましょう：
                </p>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-lg font-medium mb-2">課題シナリオ</h3>
                  <p>
                    あなたは新しいプログラミング言語を学んでいます。
                    しかし、なかなか理解が進まず、挫折しそうになっています。
                    この状況をどのように乗り越えますか？
                  </p>
                </div>
              </div>
            )}

            {/* ナビゲーションボタン */}
            <div className="mt-8 flex justify-between">
              <button
                onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
                disabled={currentStep === 1}
                className={`px-4 py-2 rounded-md ${
                  currentStep === 1
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-600 border border-gray-300 hover:bg-gray-50'
                }`}
              >
                前のステップ
              </button>
              <button
                onClick={() => setCurrentStep(Math.min(steps.length, currentStep + 1))}
                className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-500"
              >
                {currentStep === steps.length ? 'クエスト完了' : '次のステップ'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
