'use client';

import { apiUrl } from '@/lib/api';
import { DocumentTextIcon, UserIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface StudentRecord {
  id: number;
  student_name: string;
  email: string;
  class_name: string;
  last_activity: string;
  total_learning_time: number;
  quests_completed: number;
  grit_score: number;
  collaboration_score: number;
  self_regulation_score: number;
  emotional_intelligence_score: number;
}

export default function TeacherRecordsPage() {
  const [records, setRecords] = useState<StudentRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedClass, setSelectedClass] = useState<string>('all');
  const [classes, setClasses] = useState<Array<{ id: number; class_id: string; name: string }>>([]);
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
      const response = await fetch(`${apiUrl('')}/api/classes/my-classes`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setClasses(data);
        fetchRecords();
      }
    } catch (error) {
      console.error('クラス取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchRecords = async () => {
    try {
      // 実際のAPIエンドポイントに合わせて調整
      const response = await fetch(`${apiUrl('')}/api/classes/my-classes`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        // ダミーデータ（実際のAPIレスポンスに合わせて調整）
        setRecords([]);
      }
    } catch (error) {
      console.error('記録取得エラー:', error);
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
          <h1 className="text-3xl font-bold text-gray-900">生徒記録</h1>
          <p className="mt-2 text-gray-600">各生徒の詳細な学習記録と成長履歴</p>
        </div>

        {/* フィルター */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <label className="text-sm font-medium text-gray-700">クラス:</label>
            <select
              value={selectedClass}
              onChange={e => setSelectedClass(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">すべてのクラス</option>
              {classes.map(cls => (
                <option key={cls.id} value={cls.class_id}>
                  {cls.name} ({cls.class_id})
                </option>
              ))}
            </select>

            <button
              onClick={fetchRecords}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm"
            >
              更新
            </button>
          </div>
        </div>

        {/* 記録一覧 */}
        {records.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <DocumentTextIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">記録がありません</h3>
            <p className="text-gray-500 mb-6">生徒が学習を開始すると、ここに記録が表示されます</p>
            <button
              onClick={() => router.push('/teacher/classes')}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg"
            >
              クラス管理に戻る
            </button>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">生徒記録一覧</h3>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      生徒名
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      クラス
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      学習時間
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      完了クエスト
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      最終活動
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
                          <UserIcon className="h-8 w-8 text-gray-400 mr-3" />
                          <div>
                            <div className="text-sm font-medium text-gray-900">
                              {record.student_name}
                            </div>
                            <div className="text-sm text-gray-500">{record.email}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {record.class_name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {Math.round(record.total_learning_time / 60)}分
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {record.quests_completed}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {record.last_activity}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button className="text-blue-600 hover:text-blue-900">詳細を見る</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 非認知能力サマリー */}
        {records.length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">非認知能力サマリー</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {records.length > 0
                    ? Math.round(records.reduce((sum, r) => sum + r.grit_score, 0) / records.length)
                    : 0}
                  %
                </div>
                <div className="text-sm text-gray-600">平均グリット</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {records.length > 0
                    ? Math.round(
                        records.reduce((sum, r) => sum + r.collaboration_score, 0) / records.length
                      )
                    : 0}
                  %
                </div>
                <div className="text-sm text-gray-600">平均協調性</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {records.length > 0
                    ? Math.round(
                        records.reduce((sum, r) => sum + r.self_regulation_score, 0) /
                          records.length
                      )
                    : 0}
                  %
                </div>
                <div className="text-sm text-gray-600">平均自己調整</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {records.length > 0
                    ? Math.round(
                        records.reduce((sum, r) => sum + r.emotional_intelligence_score, 0) /
                          records.length
                      )
                    : 0}
                  %
                </div>
                <div className="text-sm text-gray-600">平均感情知能</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
