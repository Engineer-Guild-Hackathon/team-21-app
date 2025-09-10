'use client';

import {
  AcademicCapIcon,
  CalendarIcon,
  DocumentTextIcon,
  PencilIcon,
  PlusIcon,
  TrashIcon,
  UserIcon,
} from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface TeachingRecord {
  id: string;
  studentId: string;
  studentName: string;
  date: Date;
  subject: string;
  content: string;
  studentResponse: string;
  emotionalState: 'positive' | 'neutral' | 'negative';
  followUpActions: string[];
  teacherNotes: string;
  createdAt: Date;
  updatedAt: Date;
}

interface Student {
  id: string;
  name: string;
  grade: string;
}

export default function TeachingRecordsPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [records, setRecords] = useState<TeachingRecord[]>([]);
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingRecord, setEditingRecord] = useState<TeachingRecord | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedSubject, setSelectedSubject] = useState<string>('');
  const [recordContent, setRecordContent] = useState<string>('');
  const [studentResponse, setStudentResponse] = useState<string>('');
  const [emotionalState, setEmotionalState] = useState<'positive' | 'neutral' | 'negative'>(
    'neutral'
  );
  const [followUpActions, setFollowUpActions] = useState<string>('');
  const [teacherNotes, setTeacherNotes] = useState<string>('');

  useEffect(() => {
    if (!user) {
      router.replace('/auth/login?redirect=/records');
      return;
    }

    if (user.role !== 'teacher') {
      router.replace('/dashboard');
      return;
    }

    loadData();
  }, [user, router]);

  const loadData = async () => {
    // モックデータを読み込み
    const mockStudents: Student[] = [
      { id: '1', name: '山田太郎', grade: '5年1組' },
      { id: '2', name: '佐藤花子', grade: '5年1組' },
      { id: '3', name: '田中次郎', grade: '5年1組' },
      { id: '4', name: '鈴木三郎', grade: '5年1組' },
      { id: '5', name: '高橋四郎', grade: '5年1組' },
    ];

    const mockRecords: TeachingRecord[] = [
      {
        id: '1',
        studentId: '1',
        studentName: '山田太郎',
        date: new Date('2024-01-15'),
        subject: '数学',
        content:
          '分数の足し算について個別指導を行った。基本的な概念は理解しているが、約分のタイミングで混乱が見られた。',
        studentResponse:
          '「約分って何のタイミングでやるんですか？」と質問。具体例を使って説明すると理解が深まった。',
        emotionalState: 'positive',
        followUpActions: ['次回は分数の引き算を予定', '約分の練習問題を追加'],
        teacherNotes: '視覚的な教材を使うと理解が早い。今後も具体例を多用する。',
        createdAt: new Date('2024-01-15T14:30:00'),
        updatedAt: new Date('2024-01-15T14:30:00'),
      },
      {
        id: '2',
        studentId: '2',
        studentName: '佐藤花子',
        date: new Date('2024-01-14'),
        subject: '国語',
        content: '読解問題の解き方について指導。文章の構造を理解することの重要性を説明。',
        studentResponse: '「段落ごとに分けて考えると分かりやすい」と感想。積極的に質問してくる。',
        emotionalState: 'positive',
        followUpActions: ['読解問題集を追加で提供', '文章構造の分析練習'],
        teacherNotes: '理解力が高く、応用力もある。より高度な問題に挑戦させても良い。',
        createdAt: new Date('2024-01-14T16:00:00'),
        updatedAt: new Date('2024-01-14T16:00:00'),
      },
      {
        id: '3',
        studentId: '3',
        studentName: '田中次郎',
        date: new Date('2024-01-13'),
        subject: '数学',
        content: '小数の計算でつまずいている。位取りの概念が曖昧な様子。',
        studentResponse: '「小数点の位置が分からない」と困惑。集中力が続かない。',
        emotionalState: 'negative',
        followUpActions: ['位取りの基礎から復習', '保護者との面談を検討', '視覚教材の活用'],
        teacherNotes: '基礎的な概念の理解が不十分。段階的にアプローチする必要がある。',
        createdAt: new Date('2024-01-13T15:30:00'),
        updatedAt: new Date('2024-01-13T15:30:00'),
      },
    ];

    setStudents(mockStudents);
    setRecords(mockRecords);
    setLoading(false);
  };

  const handleAddRecord = () => {
    if (!selectedStudent || !selectedDate || !selectedSubject || !recordContent) {
      alert('必須項目を入力してください');
      return;
    }

    const student = students.find(s => s.id === selectedStudent);
    if (!student) return;

    const newRecord: TeachingRecord = {
      id: Date.now().toString(),
      studentId: selectedStudent,
      studentName: student.name,
      date: new Date(selectedDate),
      subject: selectedSubject,
      content: recordContent,
      studentResponse: studentResponse,
      emotionalState: emotionalState,
      followUpActions: followUpActions.split('\n').filter(action => action.trim()),
      teacherNotes: teacherNotes,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setRecords([newRecord, ...records]);
    resetForm();
    setShowAddModal(false);
  };

  const handleEditRecord = (record: TeachingRecord) => {
    setEditingRecord(record);
    setSelectedStudent(record.studentId);
    setSelectedDate(record.date.toISOString().split('T')[0]);
    setSelectedSubject(record.subject);
    setRecordContent(record.content);
    setStudentResponse(record.studentResponse);
    setEmotionalState(record.emotionalState);
    setFollowUpActions(record.followUpActions.join('\n'));
    setTeacherNotes(record.teacherNotes);
    setShowAddModal(true);
  };

  const handleUpdateRecord = () => {
    if (!editingRecord || !selectedStudent || !selectedDate || !selectedSubject || !recordContent) {
      alert('必須項目を入力してください');
      return;
    }

    const student = students.find(s => s.id === selectedStudent);
    if (!student) return;

    const updatedRecord: TeachingRecord = {
      ...editingRecord,
      studentId: selectedStudent,
      studentName: student.name,
      date: new Date(selectedDate),
      subject: selectedSubject,
      content: recordContent,
      studentResponse: studentResponse,
      emotionalState: emotionalState,
      followUpActions: followUpActions.split('\n').filter(action => action.trim()),
      teacherNotes: teacherNotes,
      updatedAt: new Date(),
    };

    setRecords(records.map(r => (r.id === editingRecord.id ? updatedRecord : r)));
    resetForm();
    setShowAddModal(false);
    setEditingRecord(null);
  };

  const handleDeleteRecord = (recordId: string) => {
    if (confirm('この記録を削除しますか？')) {
      setRecords(records.filter(r => r.id !== recordId));
    }
  };

  const resetForm = () => {
    setSelectedStudent('');
    setSelectedDate('');
    setSelectedSubject('');
    setRecordContent('');
    setStudentResponse('');
    setEmotionalState('neutral');
    setFollowUpActions('');
    setTeacherNotes('');
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

  const getEmotionalColor = (state: string) => {
    switch (state) {
      case 'positive':
        return 'text-green-600 bg-green-100';
      case 'neutral':
        return 'text-yellow-600 bg-yellow-100';
      case 'negative':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
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
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">指導記録</h1>
              <p className="mt-2 text-gray-600">生徒の学習指導と成長記録の管理</p>
            </div>
            <button
              onClick={() => {
                resetForm();
                setEditingRecord(null);
                setShowAddModal(true);
              }}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center"
            >
              <PlusIcon className="h-5 w-5 mr-2" />
              新規記録
            </button>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* 記録一覧 */}
        <div className="space-y-6">
          {records.map(record => (
            <div key={record.id} className="bg-white shadow rounded-lg p-6">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-4 mb-4">
                    <div className="flex items-center space-x-2">
                      <UserIcon className="h-5 w-5 text-gray-400" />
                      <span className="text-lg font-semibold text-gray-900">
                        {record.studentName}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CalendarIcon className="h-5 w-5 text-gray-400" />
                      <span className="text-sm text-gray-600">
                        {record.date.toLocaleDateString('ja-JP')}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <AcademicCapIcon className="h-5 w-5 text-gray-400" />
                      <span className="text-sm text-gray-600">{record.subject}</span>
                    </div>
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded-full ${getEmotionalColor(record.emotionalState)}`}
                    >
                      {getEmotionalIcon(record.emotionalState)}
                      {record.emotionalState === 'positive' && '良好'}
                      {record.emotionalState === 'neutral' && '普通'}
                      {record.emotionalState === 'negative' && '要注意'}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-sm font-medium text-gray-700 mb-2">指導内容</h3>
                      <p className="text-sm text-gray-900 bg-gray-50 p-3 rounded-md">
                        {record.content}
                      </p>
                    </div>

                    <div>
                      <h3 className="text-sm font-medium text-gray-700 mb-2">生徒の反応</h3>
                      <p className="text-sm text-gray-900 bg-gray-50 p-3 rounded-md">
                        {record.studentResponse}
                      </p>
                    </div>
                  </div>

                  {record.followUpActions.length > 0 && (
                    <div className="mt-4">
                      <h3 className="text-sm font-medium text-gray-700 mb-2">
                        フォローアップアクション
                      </h3>
                      <ul className="text-sm text-gray-900 bg-blue-50 p-3 rounded-md">
                        {record.followUpActions.map((action, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-blue-600 mr-2">•</span>
                            {action}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {record.teacherNotes && (
                    <div className="mt-4">
                      <h3 className="text-sm font-medium text-gray-700 mb-2">教師メモ</h3>
                      <p className="text-sm text-gray-900 bg-yellow-50 p-3 rounded-md">
                        {record.teacherNotes}
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex space-x-2 ml-4">
                  <button
                    onClick={() => handleEditRecord(record)}
                    className="text-blue-600 hover:text-blue-900"
                  >
                    <PencilIcon className="h-5 w-5" />
                  </button>
                  <button
                    onClick={() => handleDeleteRecord(record.id)}
                    className="text-red-600 hover:text-red-900"
                  >
                    <TrashIcon className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {records.length === 0 && (
          <div className="text-center py-12">
            <DocumentTextIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">記録がありません</h3>
            <p className="text-gray-600 mb-4">最初の指導記録を作成しましょう</p>
            <button
              onClick={() => {
                resetForm();
                setEditingRecord(null);
                setShowAddModal(true);
              }}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
            >
              新規記録を作成
            </button>
          </div>
        )}
      </div>

      {/* 記録追加/編集モーダル */}
      {showAddModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-full max-w-2xl shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-medium text-gray-900">
                  {editingRecord ? '記録を編集' : '新規記録を作成'}
                </h3>
                <button
                  onClick={() => {
                    setShowAddModal(false);
                    setEditingRecord(null);
                    resetForm();
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">生徒</label>
                    <select
                      value={selectedStudent}
                      onChange={e => setSelectedStudent(e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">生徒を選択</option>
                      {students.map(student => (
                        <option key={student.id} value={student.id}>
                          {student.name} ({student.grade})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">日付</label>
                    <input
                      type="date"
                      value={selectedDate}
                      onChange={e => setSelectedDate(e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">教科</label>
                  <select
                    value={selectedSubject}
                    onChange={e => setSelectedSubject(e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">教科を選択</option>
                    <option value="国語">国語</option>
                    <option value="数学">数学</option>
                    <option value="理科">理科</option>
                    <option value="社会">社会</option>
                    <option value="英語">英語</option>
                    <option value="その他">その他</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">指導内容 *</label>
                  <textarea
                    value={recordContent}
                    onChange={e => setRecordContent(e.target.value)}
                    rows={3}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="指導した内容を記録してください"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">生徒の反応</label>
                  <textarea
                    value={studentResponse}
                    onChange={e => setStudentResponse(e.target.value)}
                    rows={3}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="生徒の反応や質問を記録してください"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">感情状態</label>
                  <div className="flex space-x-4">
                    {[
                      { value: 'positive', label: '😊 良好', color: 'text-green-600' },
                      { value: 'neutral', label: '😐 普通', color: 'text-yellow-600' },
                      { value: 'negative', label: '😔 要注意', color: 'text-red-600' },
                    ].map(option => (
                      <label key={option.value} className="flex items-center">
                        <input
                          type="radio"
                          name="emotionalState"
                          value={option.value}
                          checked={emotionalState === option.value}
                          onChange={e =>
                            setEmotionalState(e.target.value as 'positive' | 'neutral' | 'negative')
                          }
                          className="mr-2"
                        />
                        <span className={option.color}>{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    フォローアップアクション
                  </label>
                  <textarea
                    value={followUpActions}
                    onChange={e => setFollowUpActions(e.target.value)}
                    rows={2}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="次回の指導計画や注意点を記録してください（1行に1つ）"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">教師メモ</label>
                  <textarea
                    value={teacherNotes}
                    onChange={e => setTeacherNotes(e.target.value)}
                    rows={2}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="その他の観察事項やメモを記録してください"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => {
                    setShowAddModal(false);
                    setEditingRecord(null);
                    resetForm();
                  }}
                  className="bg-gray-300 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-400"
                >
                  キャンセル
                </button>
                <button
                  onClick={editingRecord ? handleUpdateRecord : handleAddRecord}
                  className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
                >
                  {editingRecord ? '更新' : '作成'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
