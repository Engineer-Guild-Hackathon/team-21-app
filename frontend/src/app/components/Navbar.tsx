'use client';

import Link from 'next/link';
import { useAuth } from '../contexts/AuthContext';

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <nav className="bg-white shadow-sm fixed top-0 left-0 right-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex justify-between h-16 items-center">
          <Link href="/" className="text-xl font-bold text-gray-900">
            非認知能力学習
          </Link>

          <div className="flex items-center space-x-4">
            {user ? (
              <>
                <Link href="/learning" className="text-gray-700 hover:text-gray-900">
                  学習ページ
                </Link>
                <Link href="/progress" className="text-gray-700 hover:text-gray-900">
                  進捗確認
                </Link>
                <button onClick={logout} className="text-gray-700 hover:text-gray-900">
                  ログアウト
                </button>
              </>
            ) : (
              <>
                <Link href="/auth/login" className="text-gray-700 hover:text-gray-900">
                  ログイン
                </Link>
                <Link
                  href="/auth/register"
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                >
                  新規登録
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
