'use client';

import {
  CalendarIcon,
  ClipboardDocumentListIcon,
  EyeIcon,
  PencilIcon,
  PlusIcon,
  TrashIcon,
  UserGroupIcon,
} from '@heroicons/react/24/outline';
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function RecordsPage() {
  const { user } = useAuth();
  const [showAddForm, setShowAddForm] = useState(false);

  // 教師以外のアクセスを制限
  if (user?.role !== 'teacher') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">アクセス権限がありません</h1>
          <p className="text-gray-600">このページは教師専用です。</p>
        </div>
      </div>
    );
  }

  // デモ用の指導記録データ
  const records = [
    {
      id: 1,
      studentName: '山田太郎',
      date: '2024-01-15',
      subject: '数学',
      content: '分数の計算について指導。理解度は良好で、応用問題にも積極的に取り組んでいた。',
      observations: '集中力が高く、質問も適切だった。',
      nextSteps: '次回は小数の計算に進む予定。',
      status: 'completed',
    },
    {
      id: 2,
      studentName: '佐藤花子',
      date: '2024-01-14',
      subject: '国語',
      content: '読解力向上のための指導。文章の要点を捉える練習を行った。',
      observations: '文章の構造理解に課題がある。',
      nextSteps: '段落ごとの要点整理を重点的に指導する。',
      status: 'in_progress',
    },
    {
      id: 3,
      studentName: '田中一郎',
      date: '2024-01-13',
      subject: '理科',
      content: '植物の成長について実験観察。観察記録の書き方を指導。',
      observations: '観察力はあるが、記録の整理に課題がある。',
      nextSteps: '記録の書き方のテンプレートを提供する。',
      status: 'completed',
    },
    {
      id: 4,
      studentName: '鈴木美咲',
      date: '2024-01-12',
      subject: '社会',
      content: '歴史の年表作成。時代の流れを理解するための指導。',
      observations: '歴史への興味が高まっている。',
      nextSteps: '関連する資料を提供し、さらに深く学習させる。',
      status: 'completed',
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'in_progress':
        return 'text-yellow-600 bg-yellow-100';
      case 'pending':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed':
        return '完了';
      case 'in_progress':
        return '進行中';
      case 'pending':
        return '未着手';
      default:
        return '不明';
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">指導記録</h1>
              <p className="mt-2 text-sm text-gray-600">生徒への指導内容と観察記録を管理します</p>
            </div>
            <button
              onClick={() => setShowAddForm(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              新しい記録
            </button>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 統計カード */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 mb-8">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ClipboardDocumentListIcon className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">総記録数</dt>
                    <dd className="text-lg font-medium text-gray-900">{records.length}件</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <CalendarIcon className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">今月の記録</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {records.filter(r => r.date.startsWith('2024-01')).length}件
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <UserGroupIcon className="h-6 w-6 text-purple-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">指導対象生徒</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {new Set(records.map(r => r.studentName)).size}名
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ClipboardDocumentListIcon className="h-6 w-6 text-yellow-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">完了率</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {Math.round(
                        (records.filter(r => r.status === 'completed').length / records.length) *
                          100
                      )}
                      %
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 記録一覧 */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">指導記録一覧</h3>
            <div className="overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      生徒名
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      日付
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      科目
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      内容
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      状況
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      アクション
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {records.map(record => (
                    <tr key={record.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="h-10 w-10 flex-shrink-0">
                            <div className="h-10 w-10 rounded-full bg-blue-500 flex items-center justify-center">
                              <span className="text-white text-sm font-medium">
                                {record.studentName.charAt(0)}
                              </span>
                            </div>
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">
                              {record.studentName}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {record.date}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {record.subject}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
                        {record.content}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(record.status)}`}
                        >
                          {getStatusText(record.status)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button className="text-blue-600 hover:text-blue-900 mr-4">
                          <EyeIcon className="h-4 w-4" />
                        </button>
                        <button className="text-green-600 hover:text-green-900 mr-4">
                          <PencilIcon className="h-4 w-4" />
                        </button>
                        <button className="text-red-600 hover:text-red-900">
                          <TrashIcon className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* 新規記録追加フォーム（モーダル） */}
        {showAddForm && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
              <div className="mt-3">
                <h3 className="text-lg font-medium text-gray-900 mb-4">新しい指導記録</h3>
                <form className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">生徒名</label>
                    <select className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                      <option>山田太郎</option>
                      <option>佐藤花子</option>
                      <option>田中一郎</option>
                      <option>鈴木美咲</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">科目</label>
                    <select className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                      <option>数学</option>
                      <option>国語</option>
                      <option>理科</option>
                      <option>社会</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">指導内容</label>
                    <textarea
                      rows={3}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      placeholder="指導内容を入力してください"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">観察記録</label>
                    <textarea
                      rows={2}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      placeholder="生徒の様子や観察したことを入力してください"
                    />
                  </div>
                  <div className="flex justify-end space-x-3">
                    <button
                      type="button"
                      onClick={() => setShowAddForm(false)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300"
                    >
                      キャンセル
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
                    >
                      保存
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
