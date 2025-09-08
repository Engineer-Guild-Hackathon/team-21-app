'use client';

import {
  AcademicCapIcon,
  ChartBarIcon,
  ChatBubbleBottomCenterTextIcon,
} from '@heroicons/react/24/outline';
import { useState } from 'react';
import EmotionAnalysis from '../components/EmotionAnalysis';

export default function LearningPage() {
  const [activeTab, setActiveTab] = useState('emotion');

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">学習ダッシュボード</h1>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* タブナビゲーション */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('emotion')}
              className={`${
                activeTab === 'emotion'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              <ChartBarIcon className="mr-2 h-5 w-5" />
              感情分析
            </button>
            <button
              onClick={() => setActiveTab('quest')}
              className={`${
                activeTab === 'quest'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              <AcademicCapIcon className="mr-2 h-5 w-5" />
              学習クエスト
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`${
                activeTab === 'chat'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              <ChatBubbleBottomCenterTextIcon className="mr-2 h-5 w-5" />
              AIチャット
            </button>
          </nav>
        </div>

        {/* コンテンツエリア */}
        <div className="mt-6">
          {activeTab === 'emotion' && (
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">感情分析</h2>
                <EmotionAnalysis />
              </div>
            </div>
          )}

          {activeTab === 'quest' && (
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">学習クエスト</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {/* クエストカード */}
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      やり抜く力を育むクエスト
                    </h3>
                    <p className="text-gray-600 mb-4">
                      難しい課題に挑戦して、諦めない心を育てましょう。
                    </p>
                    <button className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-500 transition-colors">
                      開始する
                    </button>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      協調性を高めるクエスト
                    </h3>
                    <p className="text-gray-600 mb-4">
                      チームで協力して問題を解決する力を身につけましょう。
                    </p>
                    <button className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-500 transition-colors">
                      開始する
                    </button>
                  </div>

                  <div className="bg-gradient-to-br from-green-50 to-teal-50 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      創造力を伸ばすクエスト
                    </h3>
                    <p className="text-gray-600 mb-4">
                      自由な発想で新しいアイデアを生み出しましょう。
                    </p>
                    <button className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-500 transition-colors">
                      開始する
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">AIチャット</h2>
                <div className="bg-gray-50 rounded-lg p-4 mb-4 h-96 overflow-y-auto">
                  {/* チャットメッセージエリア */}
                  <div className="space-y-4">
                    <div className="flex justify-start">
                      <div className="bg-white rounded-lg p-3 shadow-sm max-w-md">
                        <p className="text-gray-900">
                          こんにちは！今日はどんなことを学びたいですか？
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="flex space-x-4">
                  <input
                    type="text"
                    placeholder="メッセージを入力..."
                    className="flex-1 rounded-lg border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  />
                  <button className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-500 transition-colors">
                    送信
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
