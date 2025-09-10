'use client';

import {
  AcademicCapIcon,
  ChartBarIcon,
  ChatBubbleBottomCenterTextIcon,
  HeartIcon,
  LightBulbIcon,
  UserGroupIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { useAuth } from './contexts/AuthContext';
const features = [
  {
    name: 'AIキャラクターとの対話',
    description: '感情を理解し、一人ひとりに寄り添った学習サポートを提供します。',
    icon: ChatBubbleBottomCenterTextIcon,
    color: 'bg-blue-500',
  },
  {
    name: '非認知能力の育成',
    description: 'やり抜く力、協調性、好奇心など、将来必要な力を楽しみながら育みます。',
    icon: LightBulbIcon,
    color: 'bg-green-500',
  },
  {
    name: 'チーム学習',
    description: '仲間と協力してミッションに挑戦し、コミュニケーション力を高めます。',
    icon: UserGroupIcon,
    color: 'bg-purple-500',
  },
  {
    name: '感情分析',
    description: 'AIが学習中の感情を理解し、最適なタイミングでサポートします。',
    icon: HeartIcon,
    color: 'bg-red-500',
  },
  {
    name: '進捗の可視化',
    description: '非認知能力の成長を分かりやすく可視化し、継続的な成長を支援します。',
    icon: ChartBarIcon,
    color: 'bg-yellow-500',
  },
  {
    name: 'アダプティブラーニング',
    description: '一人ひとりの学習スタイルに合わせて、最適な課題を提供します。',
    icon: AcademicCapIcon,
    color: 'bg-indigo-500',
  },
];

const testimonials = [
  {
    content:
      '子どもが自分から学習に取り組むようになり、驚いています。AIキャラクターとの対話が楽しいようです。',
    author: '小学4年生の保護者',
    role: '保護者',
  },
  {
    content:
      '難しい問題でも諦めずに挑戦する姿勢が身についてきました。自信もついてきているようです。',
    author: '小学5年生の担任',
    role: '教師',
  },
  {
    content: 'チームでのミッションが楽しいです。みんなで協力して問題を解くのが好きになりました。',
    author: '小学6年生',
    role: '生徒',
  },
];

export default function Home() {
  const { user, isAuthenticated } = useAuth();

  // 認証済みユーザーのダッシュボード表示
  if (isAuthenticated && user) {
    return (
      <main className="min-h-screen bg-gray-50">
        {/* ヘッダー */}
        <div className="bg-white shadow">
          <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between">
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">
                {user.role === 'student' && '学習ダッシュボード'}
                {user.role === 'parent' && '保護者ダッシュボード'}
                {user.role === 'teacher' && '教師ダッシュボード'}
              </h1>
              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-600">こんにちは、{user.name}さん</span>
                <div className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center">
                  <span className="text-white text-sm font-medium">{user.name.charAt(0)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          {/* ユーザー種別に応じたダッシュボード内容 */}
          {user.role === 'student' && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <AcademicCapIcon className="h-6 w-6 text-blue-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">今日の学習</dt>
                        <dd className="text-lg font-medium text-gray-900">3つの課題を完了</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/learning"
                      className="font-medium text-blue-600 hover:text-blue-500"
                    >
                      学習を続ける
                    </Link>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <ChartBarIcon className="h-6 w-6 text-green-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">今週の進捗</dt>
                        <dd className="text-lg font-medium text-gray-900">85% 完了</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/progress"
                      className="font-medium text-green-600 hover:text-green-500"
                    >
                      詳細を見る
                    </Link>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <ChatBubbleBottomCenterTextIcon className="h-6 w-6 text-purple-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">AIチャット</dt>
                        <dd className="text-lg font-medium text-gray-900">新しいメッセージ</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/chat"
                      className="font-medium text-purple-600 hover:text-purple-500"
                    >
                      チャットを開く
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          )}

          {user.role === 'parent' && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <ChartBarIcon className="h-6 w-6 text-blue-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">子どもの進捗</dt>
                        <dd className="text-lg font-medium text-gray-900">今週は順調に学習中</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/dashboard"
                      className="font-medium text-blue-600 hover:text-blue-500"
                    >
                      詳細を見る
                    </Link>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <HeartIcon className="h-6 w-6 text-red-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">感情分析</dt>
                        <dd className="text-lg font-medium text-gray-900">ポジティブな状態</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link href="/analysis" className="font-medium text-red-600 hover:text-red-500">
                      分析結果を見る
                    </Link>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <ChatBubbleBottomCenterTextIcon className="h-6 w-6 text-green-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">アドバイス</dt>
                        <dd className="text-lg font-medium text-gray-900">新しい提案があります</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/advice"
                      className="font-medium text-green-600 hover:text-green-500"
                    >
                      アドバイスを見る
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          )}

          {user.role === 'teacher' && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <UserGroupIcon className="h-6 w-6 text-blue-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">クラス管理</dt>
                        <dd className="text-lg font-medium text-gray-900">25名の生徒</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link href="/class" className="font-medium text-blue-600 hover:text-blue-500">
                      クラスを見る
                    </Link>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <ChartBarIcon className="h-6 w-6 text-green-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">生徒分析</dt>
                        <dd className="text-lg font-medium text-gray-900">全体の進捗を確認</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/analysis"
                      className="font-medium text-green-600 hover:text-green-500"
                    >
                      分析を見る
                    </Link>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <AcademicCapIcon className="h-6 w-6 text-purple-600" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">指導記録</dt>
                        <dd className="text-lg font-medium text-gray-900">新しい記録を追加</dd>
                      </dl>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-5 py-3">
                  <div className="text-sm">
                    <Link
                      href="/records"
                      className="font-medium text-purple-600 hover:text-purple-500"
                    >
                      記録を管理
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    );
  }

  // 未認証ユーザー向けのランディングページ
  return (
    <main className="flex min-h-screen flex-col bg-gray-50">
      {/* ヒーローセクション */}
      <div className="relative isolate overflow-hidden bg-gradient-to-r from-blue-600 to-indigo-700 pb-16 pt-14 sm:pb-20">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl py-32 sm:py-48 lg:py-56">
            <div className="text-center">
              <h1 className="text-4xl font-bold tracking-tight text-white sm:text-6xl">
                楽しみながら
                <br />
                <span className="text-blue-200">非認知能力</span>
                を育む
              </h1>
              <p className="mt-6 text-lg leading-8 text-gray-100">
                Non-Cogは、AIと感情分析を活用した次世代の学習アプリ。
                知識だけでなく、「生きる力」を育みます。
              </p>
              <div className="mt-10 flex items-center justify-center gap-x-6">
                <Link
                  href="/learning"
                  className="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-blue-600 shadow-sm hover:bg-blue-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  学習を始める
                </Link>
                <Link href="#features" className="text-sm font-semibold leading-6 text-white">
                  詳しく見る <span aria-hidden="true">→</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 特徴セクション */}
      <div id="features" className="py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-base font-semibold leading-7 text-blue-600">Non-Cogの特徴</h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              AIと感情分析で、新しい学びの形を
            </p>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              従来の学習アプリとは異なり、Non-Cogは「非認知能力」の育成に焦点を当てています。
              AIキャラクターと一緒に、楽しみながら成長できます。
            </p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3">
              {features.map(feature => (
                <div key={feature.name} className="flex flex-col">
                  <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900">
                    <div
                      className={`rounded-lg ${feature.color} p-2 ring-1 ring-inset ring-gray-200`}
                    >
                      <feature.icon className="h-5 w-5 text-white" aria-hidden="true" />
                    </div>
                    {feature.name}
                  </dt>
                  <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                    <p className="flex-auto">{feature.description}</p>
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      {/* 推薦の声セクション */}
      <div className="bg-white py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-base font-semibold leading-7 text-blue-600">利用者の声</h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              みんなの成長の物語
            </p>
          </div>
          <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-8 text-center lg:mx-0 lg:max-w-none lg:grid-cols-3">
            {testimonials.map((testimonial, index) => (
              <div
                key={index}
                className="rounded-2xl bg-gray-50 p-8 shadow-sm ring-1 ring-gray-200"
              >
                <q className="text-lg leading-8 text-gray-600">{testimonial.content}</q>
                <div className="mt-6">
                  <p className="text-base font-semibold text-gray-900">{testimonial.author}</p>
                  <p className="text-sm leading-6 text-gray-600">{testimonial.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTAセクション */}
      <div className="bg-blue-600">
        <div className="px-6 py-24 sm:px-6 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              さあ、新しい学びを始めよう
              <br />
              Non-Cogと一緒に成長しましょう
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-blue-100">
              AIキャラクターがあなたの学習をサポート。
              一人ひとりに合わせた、楽しい学習体験が待っています。
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/learning"
                className="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-blue-600 shadow-sm hover:bg-blue-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                無料で始める
              </Link>
              <Link href="#features" className="text-sm font-semibold leading-6 text-white">
                詳しく見る <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
