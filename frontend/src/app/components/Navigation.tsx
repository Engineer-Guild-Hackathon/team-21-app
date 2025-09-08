import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '../contexts/AuthContext';

const navigation = [
  { name: 'ホーム', href: '/' },
  { name: '学習', href: '/learning' },
  { name: 'フィードバック', href: '/feedback' },
  { name: '進捗', href: '/progress' },
];

const authNavigation = [
  { name: 'ログイン', href: '/auth/login' },
  { name: '新規登録', href: '/auth/register' },
];

export default function Navigation() {
  const pathname = usePathname();
  const { isAuthenticated } = useAuth();

  return (
    <nav className="flex space-x-4">
      {navigation.map((item) => {
        // 認証が必要なページへのアクセス制御
        if (!isAuthenticated && item.href !== '/') {
          return null;
        }

        // パスの一致チェック
        const isActive = pathname ? pathname.startsWith(item.href) : false;

        return (
          <Link
            key={item.name}
            href={item.href}
            className={`px-3 py-2 rounded-md text-sm font-medium ${
              isActive
                ? 'bg-gray-900 text-white'
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            {item.name}
          </Link>
        );
      })}

      {/* 認証関連のナビゲーション */}
      {!isAuthenticated &&
        authNavigation.map((item) => {
          const isActive = pathname ? pathname.startsWith(item.href) : false;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                isActive
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
            >
              {item.name}
            </Link>
          );
        })}
    </nav>
  );
}