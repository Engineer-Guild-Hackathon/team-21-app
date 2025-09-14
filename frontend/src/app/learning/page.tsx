'use client';

import { ChatBubbleLeftRightIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { geminiChatService } from '../../lib/gemini';
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
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [isAutoAnalyzing, setIsAutoAnalyzing] = useState(false);
  const [lastAnalysisTime, setLastAnalysisTime] = useState<Date | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/auth/login?redirect=/learning');
      return;
    }

    // チャットセッションを初期化または既存のセッションを読み込み
    initializeChatSession();
  }, [isAuthenticated, router]);

  // 自動分析のuseEffect
  useEffect(() => {
    if (!isAuthenticated || !currentSessionId) return;

    // 会話量に応じて分析頻度を調整
    const getAnalysisInterval = () => {
      const userMessages = messages.filter(msg => msg.role === 'user');
      if (userMessages.length >= 10) {
        return 20000; // 会話が多い場合: 20秒間隔
      } else if (userMessages.length >= 5) {
        return 30000; // 中程度の場合: 30秒間隔
      } else if (userMessages.length >= 3) {
        return 45000; // 少ない場合: 45秒間隔
      }
      return null; // 3回未満の場合は分析しない
    };

    const interval = getAnalysisInterval();
    if (!interval) return;

    const autoAnalysisInterval = setInterval(() => {
      performAutoAnalysis();
    }, interval);

    return () => clearInterval(autoAnalysisInterval);
  }, [isAuthenticated, currentSessionId, messages.length]);

  const initializeChatSession = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      // 新しいチャットセッションを作成
      const response = await fetch('http://localhost:8000/api/chat/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ title: 'AIチャット' }),
      });

      if (response.ok) {
        const session = await response.json();
        setCurrentSessionId(session.id);

        // 初期メッセージを追加
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
    } catch (error) {
      console.error('チャットセッション初期化エラー:', error);
    }
  };

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
      // ユーザーメッセージをデータベースに保存
      await saveMessageToDatabase(userMessage.content, 'user');

      // Gemini APIを使用してAI応答を取得
      const response = await geminiChatService.sendMessage(inputMessage);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);

      // AI応答をデータベースに保存
      await saveMessageToDatabase(assistantMessage.content, 'assistant');

      // ML分析を実行（会話が1回以上になったら）
      if (messages.length >= 1) {
        await analyzeConversationWithML([...messages, userMessage, assistantMessage]);
      }
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

  const saveMessageToDatabase = async (content: string, role: 'user' | 'assistant') => {
    try {
      const token = localStorage.getItem('token');
      if (!token || !currentSessionId) return;

      const response = await fetch(
        `http://localhost:8000/api/chat/sessions/${currentSessionId}/messages`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            content: content,
            role: role,
          }),
        }
      );

      if (!response.ok) {
        console.error('メッセージ保存エラー:', response.status);
      }
    } catch (error) {
      console.error('メッセージ保存エラー:', error);
    }
  };

  const analyzeConversationWithML = async (conversationMessages: Message[]) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      // 会話履歴をML分析APIに送信（バックエンドAPIの形式に合わせる）
      const messagesData = conversationMessages.map(msg => ({
        id: msg.id,
        content: msg.content,
        role: msg.role,
        timestamp: msg.timestamp.toISOString(),
      }));

      const response = await fetch('http://localhost:8000/api/ml/analyze-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(messagesData),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('ML分析結果:', result);

        // 成功通知（オプション）
        // toast.success('会話が分析されました！フィードバックページで詳細を確認できます。');
      }
    } catch (error) {
      console.error('ML分析エラー:', error);
    }
  };

  const performAutoAnalysis = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      // 既に分析中の場合はスキップ
      if (isAutoAnalyzing) return;

      setIsAutoAnalyzing(true);
      console.log('自動ML分析を実行中...');

      const response = await fetch('http://localhost:8000/api/ml/analyze-from-database', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const result = await response.json();
        console.log('自動ML分析完了:', result);
        setLastAnalysisTime(new Date());

        // 静かに成功（アラートは表示しない）
        console.log(`自動分析完了 - 会話数: ${result.conversation_count || 0}`);
        if (result.quest_data) {
          console.log(`クエスト完了数: ${result.quest_data.total_completed || 0}`);
          console.log(`連続達成日数: ${result.quest_data.max_streak_days || 0}`);
        }
      } else {
        console.error('自動ML分析エラー:', response.status);
      }
    } catch (error) {
      console.error('自動ML分析エラー:', error);
    } finally {
      setIsAutoAnalyzing(false);
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
      <div className="bg-white shadow-sm">
        <div className="mx-auto max-w-4xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <ChatBubbleLeftRightIcon className="h-6 w-6 text-indigo-600 mr-2" />
              <h1 className="text-xl font-bold text-gray-900">AIチャット</h1>
            </div>
            <p className="text-sm text-blue-600">💡 自動学習分析で成長をサポート</p>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6 lg:px-8">
        {/* チャットエリア */}
        <div className="bg-white rounded-lg shadow-lg h-[700px] flex flex-col">
          {/* ヘッダー */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center space-x-3">
              <h2 className="text-lg font-semibold text-gray-800">AI学習アシスタント</h2>
              {/* 自動分析インジケーター */}
              <div className="flex items-center space-x-2">
                {isAutoAnalyzing && (
                  <div className="flex items-center space-x-1 text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-full">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span>AIが学習を分析中...</span>
                  </div>
                )}
                {lastAnalysisTime && !isAutoAnalyzing && (
                  <div className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full">
                    ✓ 学習分析完了
                  </div>
                )}
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => router.push('/progress')}
                className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
              >
                進捗
              </button>
              <button
                onClick={() => router.push('/feedback')}
                className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600"
              >
                フィードバック
              </button>
            </div>
          </div>
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
        </div>
      </div>
    </main>
  );
}
