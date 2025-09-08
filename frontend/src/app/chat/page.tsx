'use client';

import { FaceSmileIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { useState } from 'react';

interface Message {
  id: number;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: 'こんにちは！今日はどんなことを学びたいですか？',
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [newMessage, setNewMessage] = useState('');

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return;

    // ユーザーメッセージを追加
    const userMessage: Message = {
      id: messages.length + 1,
      content: newMessage,
      sender: 'user',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');

    // AIの応答を追加（実際のAPIコールに置き換える）
    setTimeout(() => {
      const aiMessage: Message = {
        id: messages.length + 2,
        content: 'ご質問ありがとうございます。一緒に考えていきましょう！',
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, aiMessage]);
    }, 1000);
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        {/* チャットヘッダー */}
        <div className="bg-white rounded-t-lg shadow-sm p-4 border-b">
          <h1 className="text-xl font-semibold text-gray-900">AIアシスタント</h1>
          <p className="text-sm text-gray-500">24時間対応で学習をサポートします</p>
        </div>

        {/* メッセージエリア */}
        <div className="bg-white h-[calc(100vh-300px)] overflow-y-auto p-4">
          <div className="space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`rounded-lg p-4 max-w-md ${
                    message.sender === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  <p className="text-xs mt-1 opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 入力エリア */}
        <div className="bg-white rounded-b-lg shadow-sm p-4 border-t">
          <div className="flex space-x-4">
            <button
              className="p-2 text-gray-400 hover:text-gray-600"
              onClick={() => {
                /* 絵文字ピッカーを実装 */
              }}
            >
              <FaceSmileIcon className="h-6 w-6" />
            </button>
            <input
              type="text"
              value={newMessage}
              onChange={e => setNewMessage(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSendMessage()}
              placeholder="メッセージを入力..."
              className="flex-1 rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
            <button
              onClick={handleSendMessage}
              disabled={!newMessage.trim()}
              className={`p-2 rounded-lg ${
                newMessage.trim()
                  ? 'bg-blue-600 text-white hover:bg-blue-500'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
            >
              <PaperAirplaneIcon className="h-6 w-6" />
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
