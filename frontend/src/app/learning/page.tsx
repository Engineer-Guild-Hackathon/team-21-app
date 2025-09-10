'use client';

import {
  AcademicCapIcon,
  ChartBarIcon,
  ChatBubbleBottomCenterTextIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import EmotionAnalysis from '../components/EmotionAnalysis';
import { useAuth } from '../contexts/AuthContext';

export default function LearningPage() {
  const pathname = usePathname();
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!user) {
      router.replace('/auth/login?redirect=/learning');
      return;
    }
  }, [user, router]);

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

  if (!user) {
    return null;
  }

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
          <div className="bg-white rounded-lg shadow">
            <div className="p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">感情分析</h2>
              <EmotionAnalysis />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
