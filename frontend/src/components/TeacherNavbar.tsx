'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '../app/contexts/AuthContext';

export default function TeacherNavbar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const navItems = [
    { href: '/teacher/dashboard', label: 'ダッシュボード', icon: '📊' },
    { href: '/teacher/classes', label: 'クラス管理', icon: '🎓' },
    { href: '/teacher/analysis', label: '生徒分析', icon: '📈' },
    { href: '/teacher/records', label: '生徒記録', icon: '📝' },
  ];

  const isActive = (href: string) => pathname === href;

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* ロゴ・タイトル */}
          <div className="flex items-center">
            <Link href="/teacher/dashboard" className="text-xl font-bold text-gray-900">
              教師ポータル
            </Link>
          </div>

          {/* ナビゲーション */}
          <div className="hidden md:flex space-x-8">
            {navItems.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(item.href)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>

          {/* ユーザーメニュー */}
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">{user?.name || user?.email}</span>
            <button onClick={logout} className="text-sm text-gray-600 hover:text-gray-900">
              ログアウト
            </button>
          </div>
        </div>

        {/* モバイルナビゲーション */}
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navItems.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center px-3 py-2 rounded-md text-base font-medium transition-colors ${
                  isActive(item.href)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
