'use client';

import { ChartBarIcon, ClockIcon, UserGroupIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface AnalysisData {
  totalStudents: number;
  averageGritScore: number;
  averageCollaborationScore: number;
  averageSelfRegulationScore: number;
  averageEmotionalIntelligenceScore: number;
  totalLearningTime: number;
  questsCompleted: number;
}

export default function TeacherAnalysisPage() {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!user || user.role !== 'teacher') {
      router.replace('/learning');
      return;
    }
    fetchAnalysisData();
  }, [user, router]);

  const fetchAnalysisData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/classes/my-classes', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
      });
      if (response.ok) {
        const classes = await response.json();
        // 実際の分析データを取得する処理をここに追加
        // 現在はダミーデータ
        setAnalysisData({
          totalStudents: 0,
          averageGritScore: 0,
          averageCollaborationScore: 0,
          averageSelfRegulationScore: 0,
          averageEmotionalIntelligenceScore: 0,
          totalLearningTime: 0,
          questsCompleted: 0,
        });
      }
    } catch (error) {
      console.error('分析データ取得エラー:', error);
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
          <h1 className="text-3xl font-bold text-gray-900">生徒分析</h1>
          <p className="mt-2 text-gray-600">クラス全体の学習状況と非認知能力の発達を分析</p>
        </div>

        {/* 統計カード */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <UserGroupIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">総生徒数</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analysisData?.totalStudents || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <ChartBarIcon className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">完了クエスト数</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analysisData?.questsCompleted || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <ClockIcon className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">総学習時間</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analysisData?.totalLearningTime
                    ? `${Math.round(analysisData.totalLearningTime / 60)}分`
                    : '0分'}
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
                <p className="text-sm font-medium text-gray-500">平均スコア</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analysisData
                    ? Math.round(
                        (analysisData.averageGritScore +
                          analysisData.averageCollaborationScore +
                          analysisData.averageSelfRegulationScore +
                          analysisData.averageEmotionalIntelligenceScore) /
                          4
                      )
                    : 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 非認知能力分析 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">非認知能力スコア</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>グリット（やり抜く力）</span>
                  <span>{analysisData?.averageGritScore || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${analysisData?.averageGritScore || 0}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>協調性</span>
                  <span>{analysisData?.averageCollaborationScore || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${analysisData?.averageCollaborationScore || 0}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>自己調整力</span>
                  <span>{analysisData?.averageSelfRegulationScore || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full"
                    style={{ width: `${analysisData?.averageSelfRegulationScore || 0}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>感情知能</span>
                  <span>{analysisData?.averageEmotionalIntelligenceScore || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-orange-600 h-2 rounded-full"
                    style={{ width: `${analysisData?.averageEmotionalIntelligenceScore || 0}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">学習進捗</h3>
            <div className="text-center py-8">
              <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">詳細な学習進捗グラフは準備中です</p>
            </div>
          </div>
        </div>

        {/* 推奨アクション */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">推奨アクション</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">個別指導</h4>
              <p className="text-sm text-gray-600 mb-3">スコアが低い生徒への個別指導を検討</p>
              <button className="text-blue-600 text-sm font-medium">詳細を見る →</button>
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">グループ活動</h4>
              <p className="text-sm text-gray-600 mb-3">協調性向上のためのグループ活動を提案</p>
              <button className="text-blue-600 text-sm font-medium">詳細を見る →</button>
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">保護者連絡</h4>
              <p className="text-sm text-gray-600 mb-3">学習進捗について保護者に報告</p>
              <button className="text-blue-600 text-sm font-medium">詳細を見る →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
