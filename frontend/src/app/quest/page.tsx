'use client';

import {
  CheckCircleIcon,
  ClockIcon,
  LightBulbIcon,
  PencilSquareIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { useState } from 'react';

interface Step {
  id: number;
  title: string;
  description: string;
  content: {
    text: string;
    tasks?: {
      id: string;
      title: string;
      description: string;
      completed: boolean;
    }[];
    reflection?: {
      question: string;
      placeholder: string;
    };
  };
  completed: boolean;
}

export default function QuestPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [reflections, setReflections] = useState<{ [key: string]: string }>({});
  const [tasks, setTasks] = useState<{ [key: string]: boolean }>({});

  const steps: Step[] = [
    {
      id: 1,
      title: '問題を理解する',
      description: '与えられた課題の目的と要件を理解しましょう。',
      content: {
        text: 'プログラミング学習における困難な状況を想定してみましょう。以下のシナリオについて考えてみてください：',
        tasks: [
          {
            id: '1-1',
            title: '状況の把握',
            description:
              '新しい言語やフレームワークの学習で、概念が理解できず苦戦している状況です。',
            completed: false,
          },
          {
            id: '1-2',
            title: '感情の認識',
            description: 'その状況で感じる感情（焦り、不安、混乱など）を認識してください。',
            completed: false,
          },
        ],
        reflection: {
          question: 'あなたが最近直面した似たような状況について書いてみましょう。',
          placeholder: '例：ReactのHooksの概念が理解できず、焦りを感じた...',
        },
      },
      completed: false,
    },
    {
      id: 2,
      title: '解決策を考える',
      description: '複数の解決アプローチを検討し、最適な方法を選びましょう。',
      content: {
        text: '問題解決のための様々なアプローチを考えてみましょう：',
        tasks: [
          {
            id: '2-1',
            title: 'リソースの特定',
            description:
              '利用可能な学習リソース（ドキュメント、チュートリアル、コミュニティ）を列挙する。',
            completed: false,
          },
          {
            id: '2-2',
            title: '小目標の設定',
            description: '大きな課題を小さな達成可能な目標に分割する。',
            completed: false,
          },
          {
            id: '2-3',
            title: '時間管理',
            description: '各目標に適切な時間配分を設定する。',
            completed: false,
          },
        ],
        reflection: {
          question:
            'どの解決アプローチが最も効果的だと思いますか？その理由も含めて書いてください。',
          placeholder:
            '例：小さな目標に分割することで、一つずつ確実に理解を深められると考えました...',
        },
      },
      completed: false,
    },
    {
      id: 3,
      title: '実行する',
      description: '選んだ解決策を実行に移しましょう。',
      content: {
        text: '計画を実行に移す時間です。以下のステップで進めていきましょう：',
        tasks: [
          {
            id: '3-1',
            title: '環境準備',
            description: '必要な学習環境をセットアップする。',
            completed: false,
          },
          {
            id: '3-2',
            title: '基礎学習',
            description: '基本的な概念から順に学習を進める。',
            completed: false,
          },
          {
            id: '3-3',
            title: '実践練習',
            description: '学んだ内容を小さなプロジェクトで実践する。',
            completed: false,
          },
        ],
        reflection: {
          question: '実行中に直面した課題と、それをどのように乗り越えたか記録してください。',
          placeholder:
            '例：エラーメッセージの意味が分からず悩みましたが、一つずつ調べて理解を深めました...',
        },
      },
      completed: false,
    },
    {
      id: 4,
      title: '振り返る',
      description: '結果を評価し、学んだことを整理しましょう。',
      content: {
        text: '学習プロセス全体を振り返り、得られた気づきを整理しましょう：',
        tasks: [
          {
            id: '4-1',
            title: '成果の確認',
            description: '設定した目標がどの程度達成できたか確認する。',
            completed: false,
          },
          {
            id: '4-2',
            title: '学びの整理',
            description: '技術面と精神面の両方で得られた学びを整理する。',
            completed: false,
          },
        ],
        reflection: {
          question: 'この経験を通じて、あなたはどのように成長したと感じますか？',
          placeholder:
            '例：技術的な知識だけでなく、困難に立ち向かう心構えも身についたと感じます...',
        },
      },
      completed: false,
    },
  ];

  const handleTaskToggle = (taskId: string) => {
    setTasks(prev => ({
      ...prev,
      [taskId]: !prev[taskId],
    }));
  };

  const handleReflectionChange = (stepId: number, text: string) => {
    setReflections(prev => ({
      ...prev,
      [stepId]: text,
    }));
  };

  const isStepCompleted = (step: Step) => {
    const allTasksCompleted = step.content.tasks?.every(task => tasks[task.id]) ?? true;
    const reflectionCompleted = step.content.reflection ? !!reflections[step.id] : true;
    return allTasksCompleted && reflectionCompleted;
  };

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
                      ${isStepCompleted(step) ? 'bg-green-100' : currentStep === step.id ? 'bg-blue-100' : 'bg-gray-100'}
                      mx-auto
                    `}
                  >
                    {isStepCompleted(step) ? (
                      <CheckCircleIcon className="h-6 w-6 text-green-600" />
                    ) : (
                      <span
                        className={`text-lg font-semibold ${
                          currentStep === step.id ? 'text-blue-600' : 'text-gray-500'
                        }`}
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
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <LightBulbIcon className="h-6 w-6 text-yellow-500 mr-2" />
              現在のステップ: {steps[currentStep - 1]?.title}
            </h2>

            <div className="mt-6 space-y-6">
              {/* ステップの説明 */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-700">{steps[currentStep - 1]?.content.text}</p>
              </div>

              {/* タスクリスト */}
              {steps[currentStep - 1]?.content.tasks && (
                <div className="space-y-4">
                  <h3 className="text-lg font-medium flex items-center">
                    <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                    タスク
                  </h3>
                  <div className="space-y-3">
                    {steps[currentStep - 1]?.content.tasks?.map(task => (
                      <div
                        key={task.id}
                        className="flex items-start space-x-3 bg-white p-4 rounded-lg border border-gray-200"
                      >
                        <input
                          type="checkbox"
                          checked={tasks[task.id] || false}
                          onChange={() => handleTaskToggle(task.id)}
                          className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <div>
                          <p className="font-medium text-gray-900">{task.title}</p>
                          <p className="text-sm text-gray-500">{task.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 振り返り */}
              {steps[currentStep - 1]?.content.reflection && (
                <div className="space-y-4">
                  <h3 className="text-lg font-medium flex items-center">
                    <PencilSquareIcon className="h-5 w-5 text-purple-500 mr-2" />
                    振り返り
                  </h3>
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <p className="text-gray-700 mb-3">
                      {steps[currentStep - 1]?.content.reflection?.question}
                    </p>
                    <textarea
                      value={reflections[currentStep] || ''}
                      onChange={e => handleReflectionChange(currentStep, e.target.value)}
                      placeholder={steps[currentStep - 1]?.content.reflection?.placeholder}
                      rows={4}
                      className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                </div>
              )}
            </div>

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
                onClick={() => {
                  if (steps[currentStep - 1] && isStepCompleted(steps[currentStep - 1])) {
                    setCurrentStep(Math.min(steps.length, currentStep + 1));
                  }
                }}
                disabled={!steps[currentStep - 1] || !isStepCompleted(steps[currentStep - 1])}
                className={`px-4 py-2 rounded-md ${
                  !steps[currentStep - 1] || !isStepCompleted(steps[currentStep - 1])
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-500'
                }`}
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
