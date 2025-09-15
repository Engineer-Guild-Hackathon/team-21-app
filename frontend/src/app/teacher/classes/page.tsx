'use client';

import { apiUrl } from '@/lib/api';
import { PlusIcon } from '@heroicons/react/24/outline';
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

export default function TeacherClassesPage() {
  const [classes, setClasses] = useState<Class[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newClass, setNewClass] = useState({ name: '', description: '' });
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
      }
    } catch (error) {
      console.error('クラス取得エラー:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const createClass = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch(`${apiUrl('')}/api/classes/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify(newClass),
      });

      if (response.ok) {
        const createdClass = await response.json();
        setClasses([...classes, createdClass]);
        setNewClass({ name: '', description: '' });
        setShowCreateForm(false);
        alert(`クラス「${createdClass.name}」を作成しました！\nクラスID: ${createdClass.class_id}`);
      } else {
        const error = await response.json();
        alert(`エラー: ${error.detail}`);
      }
    } catch (error) {
      console.error('クラス作成エラー:', error);
      alert('クラス作成に失敗しました');
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
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">クラス管理</h1>
            <p className="mt-2 text-gray-600">作成したクラスを管理できます</p>
          </div>
          <button
            onClick={() => setShowCreateForm(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center gap-2"
          >
            <PlusIcon className="h-5 w-5" />
            新しいクラスを作成
          </button>
        </div>

        {/* クラス作成フォーム */}
        {showCreateForm && (
          <div className="bg-white rounded-lg shadow p-6 mb-8">
            <h2 className="text-xl font-semibold mb-4">新しいクラスを作成</h2>
            <form onSubmit={createClass} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">クラス名</label>
                <input
                  type="text"
                  required
                  value={newClass.name}
                  onChange={e => setNewClass({ ...newClass, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="例: 5年1組"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">説明（任意）</label>
                <textarea
                  value={newClass.description}
                  onChange={e => setNewClass({ ...newClass, description: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="クラスの説明を入力してください"
                />
              </div>
              <div className="flex gap-3">
                <button
                  type="submit"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
                >
                  クラスを作成
                </button>
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-4 py-2 rounded-md"
                >
                  キャンセル
                </button>
              </div>
            </form>
          </div>
        )}

        {/* クラス一覧 */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {classes.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <div className="text-gray-500 text-lg mb-4">まだクラスが作成されていません</div>
              <button
                onClick={() => setShowCreateForm(true)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg"
              >
                最初のクラスを作成
              </button>
            </div>
          ) : (
            classes.map(classItem => (
              <div key={classItem.id} className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">{classItem.name}</h3>
                    <p className="text-sm text-gray-500 mt-1">クラスID: {classItem.class_id}</p>
                  </div>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">
                    アクティブ
                  </span>
                </div>

                {classItem.description && (
                  <p className="text-gray-600 mb-4">{classItem.description}</p>
                )}

                <div className="flex gap-2">
                  <button
                    onClick={() => router.push(`/teacher/classes/${classItem.class_id}/students`)}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm"
                  >
                    生徒一覧を見る
                  </button>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(classItem.class_id);
                      alert('クラスIDをクリップボードにコピーしました');
                    }}
                    className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-md text-sm"
                  >
                    ID コピー
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
