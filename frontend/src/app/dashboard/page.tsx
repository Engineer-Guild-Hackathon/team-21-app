'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function DashboardPage() {
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!user) {
      router.replace('/auth/login?redirect=/dashboard');
      return;
    }
    // ロール別の最適ダッシュボードへ転送
    if (user.role === 'student') {
      // 生徒: 学習ダッシュボードへ転送
      router.replace('/learning');
      return;
    }
    if (user.role === 'teacher') {
      // 教師: クラス管理/生徒分析へ誘導
    }
  }, [user, router]);

  if (!user) {
    return null;
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">ダッシュボード</h1>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {false && (
            <Link
              href="/analysis"
              className="block rounded-lg border bg-white p-6 shadow hover:shadow-md transition"
            >
              <h2 className="text-lg font-semibold mb-2">子どもの進捗・行動分析</h2>
              <p className="text-sm text-gray-600">学習記録や感情トレンドを確認</p>
            </Link>
          )}

          {user.role === 'teacher' && (
            <>
              <Link
                href="/class"
                className="block rounded-lg border bg-white p-6 shadow hover:shadow-md transition"
              >
                <h2 className="text-lg font-semibold mb-2">クラス管理</h2>
                <p className="text-sm text-gray-600">生徒一覧や出欠・連絡</p>
              </Link>
              <Link
                href="/analysis"
                className="block rounded-lg border bg-white p-6 shadow hover:shadow-md transition"
              >
                <h2 className="text-lg font-semibold mb-2">生徒分析</h2>
                <p className="text-sm text-gray-600">学習/感情の傾向を見る</p>
              </Link>
            </>
          )}

          {user.role === 'student' && (
            <Link
              href="/learning"
              className="block rounded-lg border bg-white p-6 shadow hover:shadow-md transition"
            >
              <h2 className="text-lg font-semibold mb-2">学習ダッシュボード</h2>
              <p className="text-sm text-gray-600">学習、フィードバック、進捗へ</p>
            </Link>
          )}
        </div>
      </div>
    </main>
  );
}
