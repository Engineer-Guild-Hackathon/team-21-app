'use client';

import { FaceSmileIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { useEffect, useRef, useState } from 'react';

interface Message {
  id: number;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  emotion?: {
    type: string;
    intensity: number;
  };
}

interface AIResponse {
  message: string;
  emotion?: {
    type: string;
    intensity: number;
  };
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: 'こんにちは！今日はどんなことを学びたいですか？',
      sender: 'ai',
      timestamp: new Date(),
      emotion: {
        type: 'joy',
        intensity: 0.8,
      },
    },
  ]);
  const [newMessage, setNewMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getEmotionColor = (emotion?: { type: string; intensity: number }) => {
    if (!emotion) return 'bg-gray-100';

    const colors: { [key: string]: string } = {
      joy: 'bg-yellow-50',
      sadness: 'bg-blue-50',
      anger: 'bg-red-50',
      fear: 'bg-purple-50',
      neutral: 'bg-gray-50',
    };

    return colors[emotion.type] || 'bg-gray-50';
  };

  const generateAIResponse = async (userMessage: string): Promise<AIResponse> => {
    try {
      const response = await fetch('/api/chat/respond', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) throw new Error('AI応答の生成に失敗しました');

      return await response.json();
    } catch (error) {
      console.error('Error:', error);
      return {
        message: '申し訳ありません。一時的な問題が発生しました。',
        emotion: {
          type: 'neutral',
          intensity: 0.5,
        },
      };
    }
  };

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
    setIsTyping(true);

    // AIの応答を生成
    const aiResponse = await generateAIResponse(newMessage);

    // タイピングアニメーション用の遅延
    await new Promise(resolve => setTimeout(resolve, 1000));

    // AIの応答を追加
    const aiMessage: Message = {
      id: messages.length + 2,
      content: aiResponse.message,
      sender: 'ai',
      timestamp: new Date(),
      emotion: aiResponse.emotion,
    };
    setMessages(prev => [...prev, aiMessage]);
    setIsTyping(false);
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        {/* チャットヘッダー */}
        <div className="bg-white rounded-t-lg shadow-sm p-4 border-b">
          <h1 className="text-xl font-semibold text-gray-900">AIアシスタント</h1>
          <p className="text-sm text-gray-500">
            24時間対応で学習をサポートします。感情を理解し、適切なアドバイスを提供します。
          </p>
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
                      : `${getEmotionColor(message.emotion)} text-gray-900`
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  <p className="text-xs mt-1 opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg p-4 max-w-md">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.2s' }}
                    />
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.4s' }}
                    />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
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
              disabled={!newMessage.trim() || isTyping}
              className={`p-2 rounded-lg ${
                !newMessage.trim() || isTyping
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-500'
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
