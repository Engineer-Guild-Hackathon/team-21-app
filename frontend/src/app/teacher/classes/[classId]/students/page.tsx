'use client';

import { ArrowLeftIcon, ChartBarIcon } from '@heroicons/react/24/outline';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../../../../contexts/AuthContext';

interface LearningProgress {
  id: number;
  grit_score: number;
  collaboration_score: number;
  self_regulation_score: number;
  emotional_intelligence_score: number;
  quests_completed: number;
  total_learning_time: number;
  retry_count: number;
}

interface Student {
  student_id: number;
  student_name: string;
  student_email: string;
  progress: LearningProgress;
}

export default function ClassStudentsPage({ params }: { params: { classId: string } }) {
  const [students, setStudents] = useState<Student[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [className, setClassName] = useState('');
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!user || user.role !== 'teacher') {
      router.replace('/learning');
      return;
    }
    fetchStudents();
  }, [user, router, params.classId]);

  const fetchStudents = async () => {
    try {
      const response = await fetch(`${apiUrl('')}/api/classes/${params.classId}/students`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setStudents(data);
        setClassName(`クラス ${params.classId}`);
      } else {
        console.error('生徒一覧取得エラー');
      }
    } catch (error) {
      console.error('生徒一覧取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}時間${mins}分` : `${mins}分`;
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-lg">読み込み中...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* ヘッダー */}
        <div className="flex items-center gap-4 mb-8">
          <Link
            href="/teacher/classes"
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <ArrowLeftIcon className="h-5 w-5" />
            クラス一覧に戻る
          </Link>
        </div>

        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">{className}</h1>
          <p className="mt-2 text-gray-600">生徒の学習進捗と非認知能力スコア</p>
        </div>

        {/* 統計サマリー */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">総生徒数</p>
                <p className="text-2xl font-semibold text-gray-900">{students.length}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">平均やり抜く力</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {students.length > 0
                    ? Math.round(
                        students.reduce((sum, s) => sum + s.progress.grit_score, 0) /
                          students.length
                      )
                    : 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">平均協調性</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {students.length > 0
                    ? Math.round(
                        students.reduce((sum, s) => sum + s.progress.collaboration_score, 0) /
                          students.length
                      )
                    : 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-orange-100 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-orange-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">総学習時間</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {students.length > 0
                    ? formatTime(
                        students.reduce((sum, s) => sum + s.progress.total_learning_time, 0)
                      )
                    : '0分'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 生徒一覧 */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">生徒一覧</h2>
          </div>

          {students.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500">まだ生徒が登録されていません</p>
              <p className="text-sm text-gray-400 mt-2">
                生徒にクラスID「{params.classId}」を共有してください
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      生徒名
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      やり抜く力
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      協調性
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      自己調整
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      感情知性
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      クエスト完了
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      学習時間
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {students.map(student => (
                    <tr key={student.student_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {student.student_name}
                          </div>
                          <div className="text-sm text-gray-500">{student.student_email}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded-full ${getScoreColor(student.progress.grit_score)}`}
                        >
                          {Math.round(student.progress.grit_score)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded-full ${getScoreColor(student.progress.collaboration_score)}`}
                        >
                          {Math.round(student.progress.collaboration_score)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded-full ${getScoreColor(student.progress.self_regulation_score)}`}
                        >
                          {Math.round(student.progress.self_regulation_score)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded-full ${getScoreColor(student.progress.emotional_intelligence_score)}`}
                        >
                          {Math.round(student.progress.emotional_intelligence_score)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {student.progress.quests_completed}個
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatTime(student.progress.total_learning_time)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
