'use client';

import { apiUrl } from '@/lib/api';
import {
  AcademicCapIcon,
  ChartBarIcon,
  ClockIcon,
  UserGroupIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface Class {
  id: number;
  class_id: string;
  name: string;
  description?: string;
  teacher_id: number;
  is_active: boolean;
}

export default function TeacherDashboardPage() {
  const [classes, setClasses] = useState<Class[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!user || user.role !== 'teacher') {
      router.replace('/learning');
      return;
    }
    fetchClasses();
  }, [user, router]);

  const fetchClasses = async () => {
    try {
      const response = await fetch(apiUrl('/api/classes/my-classes'), {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setClasses(data);
      }
    } catch (error) {
      console.error('クラス取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-lg">読み込み中...</div>
      </div>
    );
  }

  return (
    <div className="py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* ヘッダー */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">教師ダッシュボード</h1>
          <p className="mt-2 text-gray-600">クラスと生徒の学習状況を管理できます</p>
        </div>

        {/* 統計カード */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <AcademicCapIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">管理クラス数</p>
                <p className="text-2xl font-semibold text-gray-900">{classes.length}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <UserGroupIcon className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">総生徒数</p>
                <p className="text-2xl font-semibold text-gray-900">0</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">平均スコア</p>
                <p className="text-2xl font-semibold text-gray-900">--</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-orange-100 rounded-lg">
                <ClockIcon className="h-6 w-6 text-orange-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">総学習時間</p>
                <p className="text-2xl font-semibold text-gray-900">--</p>
              </div>
            </div>
          </div>
        </div>

        {/* クイックアクション */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">クイックアクション</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Link
              href="/teacher/classes"
              className="flex items-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <AcademicCapIcon className="h-8 w-8 text-blue-600 mr-4" />
              <div>
                <h3 className="font-medium text-gray-900">クラス管理</h3>
                <p className="text-sm text-gray-500">クラスを作成・管理</p>
              </div>
            </Link>

            <Link
              href="/teacher/records"
              className="flex items-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <UserGroupIcon className="h-8 w-8 text-green-600 mr-4" />
              <div>
                <h3 className="font-medium text-gray-900">生徒記録</h3>
                <p className="text-sm text-gray-500">生徒の詳細記録を確認</p>
              </div>
            </Link>

            <Link
              href="/teacher/analysis"
              className="flex items-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <ChartBarIcon className="h-8 w-8 text-purple-600 mr-4" />
              <div>
                <h3 className="font-medium text-gray-900">生徒分析</h3>
                <p className="text-sm text-gray-500">詳細な学習分析</p>
              </div>
            </Link>
          </div>
        </div>

        {/* 最近のクラス */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">管理中のクラス</h2>
          </div>

          <div className="p-6">
            {classes.length === 0 ? (
              <div className="text-center py-8">
                <AcademicCapIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">まだクラスがありません</h3>
                <p className="text-gray-500 mb-6">
                  最初のクラスを作成して生徒の学習を管理しましょう
                </p>
                <Link
                  href="/teacher/classes"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg inline-flex items-center"
                >
                  <AcademicCapIcon className="h-5 w-5 mr-2" />
                  クラスを作成
                </Link>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {classes.map(classItem => (
                  <div key={classItem.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="font-medium text-gray-900">{classItem.name}</h3>
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">
                        アクティブ
                      </span>
                    </div>

                    <p className="text-sm text-gray-500 mb-3">クラスID: {classItem.class_id}</p>

                    {classItem.description && (
                      <p className="text-sm text-gray-600 mb-4">{classItem.description}</p>
                    )}

                    <Link
                      href={`/teacher/classes/${classItem.class_id}/students`}
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      生徒一覧を見る →
                    </Link>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
