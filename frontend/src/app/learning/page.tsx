'use client';

import { ChatBubbleLeftRightIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export default function AIChatPage() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/auth/login?redirect=/learning');
      return;
    }

    // 初期メッセージを追加
    if (messages.length === 0) {
      setMessages([
        {
          id: '1',
          content:
            'こんにちは！AIアシスタントです。学習について何でもお聞きください。宿題の手伝い、勉強のコツ、質問など、お気軽にどうぞ！',
          role: 'assistant',
          timestamp: new Date(),
        },
      ]);
    }
  }, [isAuthenticated, router, messages.length]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Gemini APIを使用してAI応答を取得
      const response = await geminiChatService.sendMessage(inputMessage);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('AIチャットエラー:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: '申し訳ありません。エラーが発生しました。もう一度お試しください。',
        role: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center">
            <ChatBubbleLeftRightIcon className="h-8 w-8 text-indigo-600 mr-3" />
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">AIチャット</h1>
              <p className="mt-2 text-gray-600">AIアシスタントと学習についてお話ししましょう</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6 lg:px-8">
        {/* チャットエリア */}
        <div className="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
          {/* メッセージ表示エリア */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    message.role === 'user'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                  <div
                    className={`text-xs mt-1 ${
                      message.role === 'user' ? 'text-indigo-200' : 'text-gray-500'
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}

            {/* ローディング表示 */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg px-4 py-2">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                    <span className="text-sm text-gray-600">AIが考え中...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* 入力エリア */}
          <div className="border-t border-gray-200 p-4">
            <div className="flex space-x-4">
              <textarea
                value={inputMessage}
                onChange={e => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="学習について何でも質問してください..."
                className="flex-1 resize-none rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                rows={2}
                disabled={isLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <PaperAirplaneIcon className="h-4 w-4" />
              </button>
            </div>

            {/* ヒント */}
            <div className="mt-2 text-xs text-gray-500">
              💡 ヒント: 数学の問題、宿題の手伝い、勉強のコツなどについて質問してみてください
            </div>
          </div>
        </div>

        {/* 機能説明 */}
        <div className="mt-6 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-medium text-blue-900 mb-3">
            AIチャット機能（Gemini AI搭載）
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
            <div>
              <h4 className="font-medium mb-2">📚 学習サポート</h4>
              <ul className="space-y-1">
                <li>• 数学・理科・国語・社会・英語の質問</li>
                <li>• 宿題の手伝いと問題解決</li>
                <li>• 段階的な説明と具体例</li>
                <li>• 勉強方法のアドバイス</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">🎯 非認知能力向上</h4>
              <ul className="space-y-1">
                <li>• 学習への動機づけ</li>
                <li>• 目標設定のサポート</li>
                <li>• 継続学習のコツ</li>
                <li>• グリット（やり抜く力）の育成</li>
              </ul>
            </div>
          </div>
          <div className="mt-4 p-3 bg-blue-100 rounded-lg">
            <p className="text-xs text-blue-700">
              💡 <strong>設定方法:</strong> Gemini API
              キーを設定すると、より高度なAI機能が利用できます。 環境変数{' '}
              <code>NEXT_PUBLIC_GEMINI_API_KEY</code> にAPIキーを設定してください。
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
