'use client';

import {
  ChatBubbleBottomCenterTextIcon,
  PaperAirplaneIcon,
  SparklesIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';
import { useState } from 'react';

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'ai',
      content:
        'こんにちは！私はあなたの学習をサポートするAIアシスタントです。何か困ったことや質問があれば、いつでも話しかけてくださいね！',
      timestamp: new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    // デモ用のAIレスポンス
    setTimeout(
      () => {
        const aiResponses = [
          'なるほど、それは良い質問ですね！一緒に考えてみましょう。',
          'その気持ち、とてもよく分かります。私も同じような経験があります。',
          '素晴らしい考え方ですね！その姿勢を大切にしてください。',
          '難しい問題ですね。一歩ずつ解決していきましょう。',
          'あなたの努力は必ず実を結びます。頑張ってください！',
          'その方法、とても効果的だと思います。試してみてください。',
          '困った時は、いつでも私に相談してくださいね。',
          'あなたの成長を感じられて、とても嬉しいです！',
        ];

        const randomResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)];

        const aiMessage = {
          id: messages.length + 2,
          type: 'ai',
          content: randomResponse,
          timestamp: new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' }),
        };

        setMessages(prev => [...prev, aiMessage]);
        setIsTyping(false);
      },
      1000 + Math.random() * 2000
    );
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center">
            <ChatBubbleBottomCenterTextIcon className="h-8 w-8 text-blue-600 mr-3" />
            <h1 className="text-3xl font-bold tracking-tight text-gray-900">AIチャット</h1>
          </div>
          <p className="mt-2 text-sm text-gray-600">
            AIアシスタントと対話して、学習をサポートしてもらいましょう
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
          {/* チャット履歴 */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`flex max-w-xs lg:max-w-md ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                >
                  <div className={`flex-shrink-0 ${message.type === 'user' ? 'ml-3' : 'mr-3'}`}>
                    {message.type === 'ai' ? (
                      <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                        <SparklesIcon className="h-5 w-5 text-blue-600" />
                      </div>
                    ) : (
                      <UserCircleIcon className="h-8 w-8 text-gray-400" />
                    )}
                  </div>
                  <div>
                    <div
                      className={`px-4 py-2 rounded-lg ${
                        message.type === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-900'
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                    </div>
                    <p
                      className={`text-xs text-gray-500 mt-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}
                    >
                      {message.timestamp}
                    </p>
                  </div>
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex justify-start">
                <div className="flex max-w-xs lg:max-w-md">
                  <div className="flex-shrink-0 mr-3">
                    <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                      <SparklesIcon className="h-5 w-5 text-blue-600" />
                    </div>
                  </div>
                  <div>
                    <div className="px-4 py-2 rounded-lg bg-gray-100 text-gray-900">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: '0.1s' }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: '0.2s' }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* メッセージ入力 */}
          <div className="border-t border-gray-200 p-4">
            <div className="flex space-x-4">
              <div className="flex-1">
                <textarea
                  value={inputMessage}
                  onChange={e => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="メッセージを入力してください..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 resize-none"
                  rows={2}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isTyping}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>

        {/* クイックアクション */}
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">よくある質問</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={() => setInputMessage('勉強のやる気が出ない時はどうしたらいいですか？')}
              className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <h4 className="font-medium text-gray-900">やる気が出ない時</h4>
              <p className="text-sm text-gray-600">モチベーションを上げる方法を教えて</p>
            </button>
            <button
              onClick={() => setInputMessage('友達との関係で悩んでいます')}
              className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <h4 className="font-medium text-gray-900">人間関係</h4>
              <p className="text-sm text-gray-600">友達との関係について相談したい</p>
            </button>
            <button
              onClick={() => setInputMessage('テストで緊張してしまいます')}
              className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <h4 className="font-medium text-gray-900">テスト対策</h4>
              <p className="text-sm text-gray-600">緊張を和らげる方法を知りたい</p>
            </button>
            <button
              onClick={() => setInputMessage('将来の夢について話したいです')}
              className="p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <h4 className="font-medium text-gray-900">将来の夢</h4>
              <p className="text-sm text-gray-600">将来について一緒に考えたい</p>
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
