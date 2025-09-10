'use client';

import {
  AcademicCapIcon,
  ChartBarIcon,
  ChatBubbleBottomCenterTextIcon,
  CheckCircleIcon,
  ClockIcon,
  PlayIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import EmotionAnalysis from '../components/EmotionAnalysis';
export default function LearningPage() {
  const pathname = usePathname();
  const [selectedLesson, setSelectedLesson] = useState<any>(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);

  const navigation = [
    {
      name: '感情分析',
      href: '/learning',
      icon: ChartBarIcon,
      current: pathname === '/learning',
    },
    {
      name: '学習クエスト',
      href: '/quest',
      icon: AcademicCapIcon,
      current: pathname === '/quest',
    },
    {
      name: 'AIチャット',
      href: '/chat',
      icon: ChatBubbleBottomCenterTextIcon,
      current: pathname === '/chat',
    },
  ];

  // デモ用の学習コンテンツ
  const lessons = [
    {
      id: 1,
      title: 'コミュニケーションスキル',
      description: '相手の気持ちを理解し、自分の思いを伝える方法を学びましょう',
      difficulty: '初級',
      duration: '15分',
      completed: true,
      questions: [
        {
          question: '友達が悲しそうな顔をしている時、あなたはどうしますか？',
          options: [
            '何も言わずにそっとしておく',
            '「大丈夫？」と声をかける',
            '他の友達に聞いてみる',
            '自分の話をして気をそらす',
          ],
          correct: 1,
          explanation: '友達が悲しそうな時は、優しく声をかけて話を聞いてあげることが大切です。',
        },
        {
          question: 'グループで話し合いをする時、どのように参加しますか？',
          options: [
            '最初に自分の意見を言う',
            '他の人の意見を聞いてから発言する',
            '黙って聞いているだけ',
            'リーダーに任せる',
          ],
          correct: 1,
          explanation: '他の人の意見を聞いてから発言することで、より良い話し合いができます。',
        },
      ],
    },
    {
      id: 2,
      title: '感情コントロール',
      description: '自分の感情を理解し、適切にコントロールする方法を学びましょう',
      difficulty: '中級',
      duration: '20分',
      completed: false,
      questions: [
        {
          question: 'テストで悪い点を取ってしまった時、どう対処しますか？',
          options: [
            '落ち込んで何もしたくなくなる',
            '次回頑張ろうと思う',
            '先生に文句を言う',
            '友達の点数と比べる',
          ],
          correct: 1,
          explanation: '失敗を次への学習機会として捉えることで、成長につなげることができます。',
        },
      ],
    },
    {
      id: 3,
      title: 'チームワーク',
      description: 'みんなで協力して目標を達成する方法を学びましょう',
      difficulty: '中級',
      duration: '25分',
      completed: false,
      questions: [
        {
          question: 'グループ活動で意見が分かれた時、どうしますか？',
          options: [
            '自分の意見を通す',
            'みんなで話し合って決める',
            'リーダーに任せる',
            '投票で決める',
          ],
          correct: 1,
          explanation: 'みんなで話し合うことで、全員が納得できる解決策を見つけることができます。',
        },
      ],
    },
  ];

  const handleStartLesson = (lesson: any) => {
    setSelectedLesson(lesson);
    setCurrentQuestion(0);
    setScore(0);
    setShowResult(false);
  };

  const handleAnswer = (answerIndex: number) => {
    if (selectedLesson && currentQuestion < selectedLesson.questions.length) {
      const question = selectedLesson.questions[currentQuestion];
      if (answerIndex === question.correct) {
        setScore(score + 1);
      }

      if (currentQuestion + 1 < selectedLesson.questions.length) {
        setCurrentQuestion(currentQuestion + 1);
      } else {
        setShowResult(true);
      }
    }
  };

  const resetLesson = () => {
    setSelectedLesson(null);
    setCurrentQuestion(0);
    setScore(0);
    setShowResult(false);
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">学習ダッシュボード</h1>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* ナビゲーション */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            {navigation.map(item => (
              <Link
                key={item.name}
                href={item.href}
                className={`${
                  item.current
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
              >
                <item.icon className="mr-2 h-5 w-5" />
                {item.name}
              </Link>
            ))}
          </nav>
        </div>

        {/* メインコンテンツ */}
        <div className="mt-6">
          {!selectedLesson ? (
            <div className="space-y-6">
              {/* 学習レッスン一覧 */}
              <div className="bg-white rounded-lg shadow">
                <div className="p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">学習レッスン</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {lessons.map(lesson => (
                      <div
                        key={lesson.id}
                        className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <h3 className="text-lg font-medium text-gray-900">{lesson.title}</h3>
                          {lesson.completed && (
                            <CheckCircleIcon className="h-5 w-5 text-green-500" />
                          )}
                        </div>
                        <p className="text-sm text-gray-600 mb-3">{lesson.description}</p>
                        <div className="flex items-center justify-between mb-3">
                          <span
                            className={`px-2 py-1 text-xs font-medium rounded-full ${
                              lesson.difficulty === '初級'
                                ? 'bg-green-100 text-green-800'
                                : lesson.difficulty === '中級'
                                  ? 'bg-yellow-100 text-yellow-800'
                                  : 'bg-red-100 text-red-800'
                            }`}
                          >
                            {lesson.difficulty}
                          </span>
                          <div className="flex items-center text-sm text-gray-500">
                            <ClockIcon className="h-4 w-4 mr-1" />
                            {lesson.duration}
                          </div>
                        </div>
                        <button
                          onClick={() => handleStartLesson(lesson)}
                          className="w-full flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          <PlayIcon className="h-4 w-4 mr-2" />
                          {lesson.completed ? '再挑戦' : '開始'}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 感情分析 */}
              <div className="bg-white rounded-lg shadow">
                <div className="p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">感情分析</h2>
                  <EmotionAnalysis />
                </div>
              </div>
            </div>
          ) : !showResult ? (
            /* クイズ画面 */
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-medium text-gray-900">{selectedLesson.title}</h2>
                  <div className="text-sm text-gray-500">
                    問題 {currentQuestion + 1} / {selectedLesson.questions.length}
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="text-xl font-medium text-gray-900 mb-4">
                    {selectedLesson.questions[currentQuestion].question}
                  </h3>
                  <div className="space-y-3">
                    {selectedLesson.questions[currentQuestion].options.map(
                      (option: string, index: number) => (
                        <button
                          key={index}
                          onClick={() => handleAnswer(index)}
                          className="w-full text-left p-4 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
                        >
                          {option}
                        </button>
                      )
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* 結果画面 */
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <div className="text-center">
                  <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
                    <StarIcon className="h-6 w-6 text-green-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">お疲れ様でした！</h2>
                  <p className="text-lg text-gray-600 mb-4">{selectedLesson.title}を完了しました</p>
                  <div className="text-3xl font-bold text-blue-600 mb-4">
                    {score} / {selectedLesson.questions.length} 問正解
                  </div>
                  <div className="text-sm text-gray-500 mb-6">
                    正解率: {Math.round((score / selectedLesson.questions.length) * 100)}%
                  </div>
                  <div className="space-x-4">
                    <button
                      onClick={resetLesson}
                      className="px-6 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      他のレッスンに戻る
                    </button>
                    <button
                      onClick={() => handleStartLesson(selectedLesson)}
                      className="px-6 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      もう一度挑戦
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
