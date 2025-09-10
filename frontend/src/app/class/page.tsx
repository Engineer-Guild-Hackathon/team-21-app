'use client';

import {
  ChartBarIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  UserGroupIcon,
} from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface Student {
  id: string;
  name: string;
  grade: string;
  attendance: 'present' | 'absent' | 'late';
  lastLogin: Date;
  learningProgress: number;
  emotionalState: 'positive' | 'neutral' | 'negative';
  recentActivities: number;
  concerns: string[];
}

interface ClassStats {
  totalStudents: number;
  presentToday: number;
  averageProgress: number;
  studentsWithConcerns: number;
  recentActivities: number;
}

export default function ClassManagementPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [students, setStudents] = useState<Student[]>([]);
  const [classStats, setClassStats] = useState<ClassStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);

  useEffect(() => {
    if (!user) {
      router.replace('/auth/login?redirect=/class');
      return;
    }

    if (user.role !== 'teacher') {
      router.replace('/dashboard');
      return;
    }

    loadClassData();
  }, [user, router]);

  const loadClassData = async () => {
    // モックデータを読み込み
    const mockStudents: Student[] = [
      {
        id: '1',
        name: '山田太郎',
        grade: '5年1組',
        attendance: 'present',
        lastLogin: new Date('2024-01-15T09:30:00'),
        learningProgress: 85,
        emotionalState: 'positive',
        recentActivities: 5,
        concerns: [],
      },
      {
        id: '2',
        name: '佐藤花子',
        grade: '5年1組',
        attendance: 'present',
        lastLogin: new Date('2024-01-15T10:15:00'),
        learningProgress: 92,
        emotionalState: 'positive',
        recentActivities: 7,
        concerns: [],
      },
      {
        id: '3',
        name: '田中次郎',
        grade: '5年1組',
        attendance: 'late',
        lastLogin: new Date('2024-01-15T11:00:00'),
        learningProgress: 65,
        emotionalState: 'neutral',
        recentActivities: 3,
        concerns: ['数学の理解が遅れている'],
      },
      {
        id: '4',
        name: '鈴木三郎',
        grade: '5年1組',
        attendance: 'absent',
        lastLogin: new Date('2024-01-14T16:30:00'),
        learningProgress: 45,
        emotionalState: 'negative',
        recentActivities: 1,
        concerns: ['学習意欲の低下', '集中力の不足'],
      },
      {
        id: '5',
        name: '高橋四郎',
        grade: '5年1組',
        attendance: 'present',
        lastLogin: new Date('2024-01-15T08:45:00'),
        learningProgress: 78,
        emotionalState: 'positive',
        recentActivities: 6,
        concerns: [],
      },
    ];

    const mockStats: ClassStats = {
      totalStudents: mockStudents.length,
      presentToday: mockStudents.filter(s => s.attendance === 'present').length,
      averageProgress: Math.round(
        mockStudents.reduce((sum, s) => sum + s.learningProgress, 0) / mockStudents.length
      ),
      studentsWithConcerns: mockStudents.filter(s => s.concerns.length > 0).length,
      recentActivities: mockStudents.reduce((sum, s) => sum + s.recentActivities, 0),
    };

    setStudents(mockStudents);
    setClassStats(mockStats);
    setLoading(false);
  };

  const getAttendanceColor = (attendance: string) => {
    switch (attendance) {
      case 'present':
        return 'bg-green-100 text-green-800';
      case 'late':
        return 'bg-yellow-100 text-yellow-800';
      case 'absent':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getAttendanceText = (attendance: string) => {
    switch (attendance) {
      case 'present':
        return '出席';
      case 'late':
        return '遅刻';
      case 'absent':
        return '欠席';
      default:
        return '不明';
    }
  };

  const getEmotionalIcon = (state: string) => {
    switch (state) {
      case 'positive':
        return '😊';
      case 'neutral':
        return '😐';
      case 'negative':
        return '😔';
      default:
        return '❓';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">データを読み込み中...</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">クラス管理</h1>
              <p className="mt-2 text-gray-600">5年1組 - 生徒の学習状況と出席管理</p>
            </div>
            <div className="flex space-x-4">
              <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                出席記録
              </button>
              <button className="bg-gray-200 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-300">
                クラスレポート
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* クラス統計 */}
        {classStats && (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <UserGroupIcon className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">総生徒数</p>
                  <p className="text-2xl font-bold text-gray-900">{classStats.totalStudents}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <CheckCircleIcon className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">今日の出席</p>
                  <p className="text-2xl font-bold text-gray-900">{classStats.presentToday}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <ChartBarIcon className="h-8 w-8 text-purple-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">平均進捗</p>
                  <p className="text-2xl font-bold text-gray-900">{classStats.averageProgress}%</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <ExclamationTriangleIcon className="h-8 w-8 text-yellow-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">要注意生徒</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {classStats.studentsWithConcerns}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <ClockIcon className="h-8 w-8 text-indigo-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">今日の活動</p>
                  <p className="text-2xl font-bold text-gray-900">{classStats.recentActivities}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 生徒一覧 */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">生徒一覧</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    生徒名
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    出席状況
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    学習進捗
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    感情状態
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    最終ログイン
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    アクション
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {students.map(student => (
                  <tr key={student.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10">
                          <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                            <span className="text-sm font-medium text-blue-600">
                              {student.name.charAt(0)}
                            </span>
                          </div>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">{student.name}</div>
                          <div className="text-sm text-gray-500">{student.grade}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getAttendanceColor(student.attendance)}`}
                      >
                        {getAttendanceText(student.attendance)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${student.learningProgress}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900">{student.learningProgress}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <span className="text-lg mr-2">
                          {getEmotionalIcon(student.emotionalState)}
                        </span>
                        <span className="text-sm text-gray-900">
                          {student.emotionalState === 'positive' && '良好'}
                          {student.emotionalState === 'neutral' && '普通'}
                          {student.emotionalState === 'negative' && '要注意'}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {student.lastLogin.toLocaleDateString('ja-JP')}{' '}
                      {student.lastLogin.toLocaleTimeString('ja-JP', {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => setSelectedStudent(student)}
                        className="text-blue-600 hover:text-blue-900 mr-4"
                      >
                        詳細
                      </button>
                      <button className="text-green-600 hover:text-green-900">連絡</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 生徒詳細モーダル */}
        {selectedStudent && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
              <div className="mt-3">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    {selectedStudent.name} の詳細
                  </h3>
                  <button
                    onClick={() => setSelectedStudent(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    ✕
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">学習進捗</label>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${selectedStudent.learningProgress}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {selectedStudent.learningProgress}%
                      </p>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">感情状態</label>
                    <p className="mt-1 text-sm text-gray-900">
                      {getEmotionalIcon(selectedStudent.emotionalState)}
                      {selectedStudent.emotionalState === 'positive' && ' 良好'}
                      {selectedStudent.emotionalState === 'neutral' && ' 普通'}
                      {selectedStudent.emotionalState === 'negative' && ' 要注意'}
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">最近の活動数</label>
                    <p className="mt-1 text-sm text-gray-900">
                      {selectedStudent.recentActivities}件
                    </p>
                  </div>

                  {selectedStudent.concerns.length > 0 && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700">注意事項</label>
                      <ul className="mt-1 text-sm text-gray-900 list-disc list-inside">
                        {selectedStudent.concerns.map((concern, index) => (
                          <li key={index}>{concern}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    onClick={() => setSelectedStudent(null)}
                    className="bg-gray-300 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-400"
                  >
                    閉じる
                  </button>
                  <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                    詳細分析
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
