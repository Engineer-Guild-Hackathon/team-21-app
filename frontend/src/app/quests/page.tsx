'use client';

import { useAuth } from '@/app/contexts/AuthContext';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { BookOpen, Clock, Lightbulb, Star, Trophy, Users } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

interface Quest {
  id: number;
  title: string;
  description: string;
  quest_type: string;
  difficulty: string;
  target_skill: string;
  estimated_duration: number;
  experience_points: number;
  coins: number;
  is_daily: boolean;
}

interface QuestProgress {
  id: number;
  status: string;
  progress_percentage: number;
  current_step: number;
  total_steps: number;
  quest: Quest;
}

interface QuestStats {
  total_quests: number;
  completed_quests: number;
  in_progress_quests: number;
  total_experience: number;
  total_coins: number;
  streak_days: number;
  favorite_quest_type: string | null;
}

const questTypeIcons = {
  daily_log: BookOpen,
  plant_care: Star,
  story_creation: Lightbulb,
  collaboration: Users,
};

const questTypeLabels = {
  daily_log: '今日の冒険ログ',
  plant_care: '魔法の種を育てよう',
  story_creation: 'お話の森',
  collaboration: '協力！謎解きチャットルーム',
};

const difficultyColors = {
  easy: 'bg-green-100 text-green-800',
  medium: 'bg-yellow-100 text-yellow-800',
  hard: 'bg-red-100 text-red-800',
};

const difficultyLabels = {
  easy: '簡単',
  medium: '普通',
  hard: '難しい',
};

const statusColors = {
  not_started: 'bg-gray-100 text-gray-800',
  in_progress: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  locked: 'bg-gray-100 text-gray-600',
};

const statusLabels = {
  not_started: '未開始',
  in_progress: '進行中',
  completed: '完了',
  locked: 'ロック中',
};

export default function QuestsPage() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [quests, setQuests] = useState<Quest[]>([]);
  const [progress, setProgress] = useState<QuestProgress[]>([]);
  const [stats, setStats] = useState<QuestStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'available' | 'progress' | 'stats'>('available');

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/auth/login');
      return;
    }

    fetchQuests();
    fetchProgress();
    fetchStats();
  }, [isAuthenticated, router]);

  const fetchQuests = async () => {
    try {
      const token = localStorage.getItem('token');
      console.log('Token:', token);

      const response = await fetch('http://localhost:8000/api/quests/', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (response.ok) {
        const data = await response.json();
        console.log('Quests data:', data);
        setQuests(data.quests);
      } else {
        const errorText = await response.text();
        console.error('Response error:', errorText);
      }
    } catch (error) {
      console.error('クエスト取得エラー:', error);
    }
  };

  const fetchProgress = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/quests/my-progress', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setProgress(data.progress);
      }
    } catch (error) {
      console.error('進捗取得エラー:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/quests/stats', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('統計取得エラー:', error);
    } finally {
      setLoading(false);
    }
  };

  const startQuest = async (questId: number) => {
    try {
      const response = await fetch(`http://localhost:8000/api/quests/${questId}/start`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
      });
      if (response.ok) {
        fetchProgress();
        fetchStats();
      }
    } catch (error) {
      console.error('クエスト開始エラー:', error);
    }
  };

  const getQuestProgress = (questId: number) => {
    return progress.find(p => p.quest.id === questId);
  };

  const getQuestIcon = (questType: string) => {
    const IconComponent = questTypeIcons[questType as keyof typeof questTypeIcons] || BookOpen;
    return <IconComponent className="w-6 h-6" />;
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg">読み込み中...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* ヘッダー */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">課題問題クエスト</h1>
        <p className="text-gray-600">
          様々な課題問題にチャレンジして、スキルアップを目指しましょう！
        </p>
      </div>

      {/* タブ */}
      <div className="mb-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('available')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'available'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              利用可能なクエスト
            </button>
            <button
              onClick={() => setActiveTab('progress')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'progress'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              進行中のクエスト
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'stats'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              統計情報
            </button>
          </nav>
        </div>
      </div>

      {/* コンテンツ */}
      {activeTab === 'available' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {quests.map(quest => {
            const questProgress = getQuestProgress(quest.id);
            const canStart = !questProgress || questProgress.status === 'not_started';

            return (
              <Card key={quest.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getQuestIcon(quest.quest_type)}
                      <Badge
                        className={
                          difficultyColors[quest.difficulty as keyof typeof difficultyColors]
                        }
                      >
                        {difficultyLabels[quest.difficulty as keyof typeof difficultyLabels]}
                      </Badge>
                    </div>
                    {quest.is_daily && (
                      <Badge variant="outline" className="text-orange-600 border-orange-600">
                        日次
                      </Badge>
                    )}
                  </div>
                  <CardTitle className="text-lg">{quest.title}</CardTitle>
                  <CardDescription>{quest.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm text-gray-600">
                      <span>対象スキル: {quest.target_skill}</span>
                    </div>

                    <div className="flex items-center space-x-4 text-sm">
                      <div className="flex items-center space-x-1">
                        <Clock className="w-4 h-4" />
                        <span>{quest.estimated_duration}分</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Star className="w-4 h-4 text-yellow-500" />
                        <span>{quest.experience_points}XP</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Trophy className="w-4 h-4 text-yellow-600" />
                        <span>{quest.coins}コイン</span>
                      </div>
                    </div>

                    {questProgress && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>進捗</span>
                          <span>{questProgress.progress_percentage.toFixed(0)}%</span>
                        </div>
                        <Progress value={questProgress.progress_percentage} className="h-2" />
                        <Badge
                          className={
                            statusColors[questProgress.status as keyof typeof statusColors]
                          }
                        >
                          {statusLabels[questProgress.status as keyof typeof statusLabels]}
                        </Badge>
                      </div>
                    )}

                    <Button
                      onClick={() => startQuest(quest.id)}
                      disabled={!canStart}
                      className="w-full"
                      variant={canStart ? 'default' : 'outline'}
                    >
                      {canStart ? 'クエスト開始' : '進行中'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {activeTab === 'progress' && (
        <div className="space-y-6">
          {progress.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <BookOpen className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  進行中のクエストはありません
                </h3>
                <p className="text-gray-600">新しいクエストを開始してみましょう！</p>
              </CardContent>
            </Card>
          ) : (
            progress.map(questProgress => (
              <Card key={questProgress.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getQuestIcon(questProgress.quest.quest_type)}
                      <CardTitle>{questProgress.quest.title}</CardTitle>
                    </div>
                    <Badge
                      className={statusColors[questProgress.status as keyof typeof statusColors]}
                    >
                      {statusLabels[questProgress.status as keyof typeof statusLabels]}
                    </Badge>
                  </div>
                  <CardDescription>{questProgress.quest.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between text-sm">
                      <span>進捗</span>
                      <span>{questProgress.progress_percentage.toFixed(0)}%</span>
                    </div>
                    <Progress value={questProgress.progress_percentage} className="h-3" />

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">ステップ:</span>
                        <span className="ml-2 font-medium">
                          {questProgress.current_step} / {questProgress.total_steps}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">対象スキル:</span>
                        <span className="ml-2 font-medium">{questProgress.quest.target_skill}</span>
                      </div>
                    </div>

                    {questProgress.status === 'in_progress' && (
                      <Button
                        onClick={() => router.push(`/quests/${questProgress.quest.id}`)}
                        className="w-full"
                      >
                        クエストを続ける
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      )}

      {activeTab === 'stats' && stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">総クエスト数</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_quests}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">完了クエスト</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.completed_quests}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">獲得経験値</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">{stats.total_experience}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">獲得コイン</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">{stats.total_coins}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">連続達成日数</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{stats.streak_days}日</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">お気に入りタイプ</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-medium">
                {stats.favorite_quest_type
                  ? questTypeLabels[stats.favorite_quest_type as keyof typeof questTypeLabels]
                  : 'なし'}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
