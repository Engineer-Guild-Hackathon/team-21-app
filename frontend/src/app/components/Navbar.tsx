'use client';

import { Disclosure, Menu, Transition } from '@headlessui/react';
import {
  AcademicCapIcon,
  Bars3Icon,
  ChartBarIcon,
  ChatBubbleLeftRightIcon,
  ClipboardDocumentListIcon,
  HomeIcon,
  UserCircleIcon,
  UserGroupIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Fragment } from 'react';
import { useAuth, UserRole } from '../contexts/AuthContext';

// ユーザー種別ごとのナビゲーション項目
const navigationByRole: Record<UserRole, Array<{ name: string; href: string; icon: any }>> = {
  student: [
    { name: 'ホーム', href: '/', icon: HomeIcon },
    { name: '学習', href: '/learning', icon: AcademicCapIcon },
    { name: 'フィードバック', href: '/feedback', icon: ChatBubbleLeftRightIcon },
    { name: '進捗', href: '/progress', icon: ChartBarIcon },
  ],
  parent: [
    { name: 'ホーム', href: '/', icon: HomeIcon },
    { name: '子どもの進捗', href: '/dashboard', icon: ChartBarIcon },
    { name: '行動分析', href: '/analysis', icon: ClipboardDocumentListIcon },
    { name: 'アドバイス', href: '/advice', icon: ChatBubbleLeftRightIcon },
  ],
  teacher: [
    { name: 'ホーム', href: '/', icon: HomeIcon },
    { name: 'クラス管理', href: '/class', icon: UserGroupIcon },
    { name: '生徒分析', href: '/analysis', icon: ChartBarIcon },
    { name: '指導記録', href: '/records', icon: ClipboardDocumentListIcon },
  ],
};

// ユーザー種別ごとのメニュー項目
const menuItemsByRole: Record<UserRole, Array<{ name: string; href: string }>> = {
  student: [
    { name: 'プロフィール', href: '/profile' },
    { name: '設定', href: '/settings' },
  ],
  parent: [
    { name: '子ども管理', href: '/children' },
    { name: '通知設定', href: '/notifications' },
    { name: '設定', href: '/settings' },
  ],
  teacher: [
    { name: 'クラス設定', href: '/class/settings' },
    { name: '教材管理', href: '/materials' },
    { name: '設定', href: '/settings' },
  ],
};

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

export default function Navbar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  // ユーザーの種類に応じたナビゲーション項目を取得
  const navigation = user ? navigationByRole[user.role] : navigationByRole.student;
  const menuItems = user ? menuItemsByRole[user.role] : [];

  return (
    <Disclosure as="nav" className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg">
      {({ open }) => (
        <>
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 justify-between">
              <div className="flex">
                <div className="flex flex-shrink-0 items-center">
                  <span className="text-2xl font-bold text-white">Non-Cog</span>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  {navigation.map(item => (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={classNames(
                        pathname === item.href
                          ? 'border-white text-white'
                          : 'border-transparent text-gray-200 hover:border-gray-300 hover:text-white',
                        'inline-flex items-center border-b-2 px-1 pt-1 text-sm font-medium'
                      )}
                    >
                      <item.icon className="h-5 w-5 mr-2" />
                      {item.name}
                    </Link>
                  ))}
                </div>
              </div>

              <div className="hidden sm:ml-6 sm:flex sm:items-center">
                {user ? (
                  <Menu as="div" className="relative ml-3">
                    <div>
                      <Menu.Button className="flex rounded-full bg-white p-1 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                        {user.avatar ? (
                          <img className="h-8 w-8 rounded-full" src={user.avatar} alt={user.name} />
                        ) : (
                          <UserCircleIcon className="h-6 w-6 text-gray-700" aria-hidden="true" />
                        )}
                      </Menu.Button>
                    </div>
                    <Transition
                      as={Fragment}
                      enter="transition ease-out duration-200"
                      enterFrom="transform opacity-0 scale-95"
                      enterTo="transform opacity-100 scale-100"
                      leave="transition ease-in duration-75"
                      leaveFrom="transform opacity-100 scale-100"
                      leaveTo="transform opacity-0 scale-95"
                    >
                      <Menu.Items className="absolute right-0 z-10 mt-2 w-48 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                        <div className="px-4 py-2 text-sm text-gray-500">
                          {user.name}
                          <div className="text-xs text-gray-400">
                            {user.role === 'student' && '生徒'}
                            {user.role === 'parent' && '保護者'}
                            {user.role === 'teacher' && '教師'}
                          </div>
                        </div>
                        {menuItems.map(item => (
                          <Menu.Item key={item.name}>
                            {({ active }) => (
                              <Link
                                href={item.href}
                                className={classNames(
                                  active ? 'bg-gray-100' : '',
                                  'block px-4 py-2 text-sm text-gray-700'
                                )}
                              >
                                {item.name}
                              </Link>
                            )}
                          </Menu.Item>
                        ))}
                        <Menu.Item>
                          {({ active }) => (
                            <button
                              onClick={logout}
                              className={classNames(
                                active ? 'bg-gray-100' : '',
                                'block w-full px-4 py-2 text-left text-sm text-gray-700'
                              )}
                            >
                              ログアウト
                            </button>
                          )}
                        </Menu.Item>
                      </Menu.Items>
                    </Transition>
                  </Menu>
                ) : (
                  <div className="space-x-4">
                    <Link
                      href="/auth/login"
                      className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-50"
                    >
                      ログイン
                    </Link>
                    <Link
                      href="/auth/register"
                      className="rounded-md bg-indigo-500 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400"
                    >
                      新規登録
                    </Link>
                  </div>
                )}
              </div>

              <div className="-mr-2 flex items-center sm:hidden">
                <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-gray-200 hover:bg-indigo-500 hover:text-white focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                  <span className="sr-only">メニューを開く</span>
                  {open ? (
                    <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
                  ) : (
                    <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
                  )}
                </Disclosure.Button>
              </div>
            </div>
          </div>

          <Disclosure.Panel className="sm:hidden">
            <div className="space-y-1 pb-3 pt-2">
              {navigation.map(item => (
                <Disclosure.Button
                  key={item.name}
                  as={Link}
                  href={item.href}
                  className={classNames(
                    pathname === item.href
                      ? 'bg-indigo-50 border-indigo-500 text-indigo-700'
                      : 'border-transparent text-gray-200 hover:bg-indigo-500 hover:text-white',
                    'block border-l-4 py-2 pl-3 pr-4 text-base font-medium'
                  )}
                >
                  <div className="flex items-center">
                    <item.icon className="h-5 w-5 mr-2" />
                    {item.name}
                  </div>
                </Disclosure.Button>
              ))}
            </div>
            {user ? (
              <div className="border-t border-gray-200 pb-3 pt-4">
                <div className="px-4 py-2">
                  <div className="text-base font-medium text-gray-200">{user.name}</div>
                  <div className="text-sm text-gray-400">{user.email}</div>
                </div>
                <div className="space-y-1">
                  {menuItems.map(item => (
                    <Disclosure.Button
                      key={item.name}
                      as={Link}
                      href={item.href}
                      className="block px-4 py-2 text-base font-medium text-gray-200 hover:bg-indigo-500 hover:text-white"
                    >
                      {item.name}
                    </Disclosure.Button>
                  ))}
                  <Disclosure.Button
                    onClick={logout}
                    className="block w-full px-4 py-2 text-left text-base font-medium text-gray-200 hover:bg-indigo-500 hover:text-white"
                  >
                    ログアウト
                  </Disclosure.Button>
                </div>
              </div>
            ) : (
              <div className="border-t border-gray-200 pb-3 pt-4">
                <div className="space-y-1">
                  <Disclosure.Button
                    as={Link}
                    href="/auth/login"
                    className="block px-4 py-2 text-base font-medium text-gray-200 hover:bg-indigo-500 hover:text-white"
                  >
                    ログイン
                  </Disclosure.Button>
                  <Disclosure.Button
                    as={Link}
                    href="/auth/register"
                    className="block px-4 py-2 text-base font-medium text-gray-200 hover:bg-indigo-500 hover:text-white"
                  >
                    新規登録
                  </Disclosure.Button>
                </div>
              </div>
            )}
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}
