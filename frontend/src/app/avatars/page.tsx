'use client';

import { useAuth } from '@/app/contexts/AuthContext';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { SparklesIcon, TrophyIcon, UserCircleIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

interface Avatar {
  id: number;
  name: string;
  description: string;
  image_url: string;
  category: string;
  rarity: string;
  unlock_condition_type: string;
  unlock_condition_value: number;
}

interface Title {
  id: number;
  name: string;
  description: string;
  icon_url: string;
  category: string;
  rarity: string;
  unlock_condition_type: string;
  unlock_condition_value: number;
  unlock_condition_description: string;
}

interface UserAvatar {
  id: number;
  avatar_id: number;
  is_current: boolean;
  unlocked_at: string;
  avatar: Avatar;
}

interface UserTitle {
  id: number;
  title_id: number;
  is_current: boolean;
  unlocked_at: string;
  title: Title;
}

interface UserStats {
  id: number;
  user_id: number;
  total_quests_completed: number;
  daily_quests_completed: number;
  current_streak_days: number;
  max_streak_days: number;
  total_learning_time_minutes: number;
  total_sessions: number;
  grit_level: number;
  collaboration_level: number;
  self_regulation_level: number;
  emotional_intelligence_level: number;
  total_titles_earned: number;
  total_avatars_unlocked: number;
}

interface UserProfile {
  id: number;
  name: string;
  email: string;
  role: string;
  current_avatar: UserAvatar | null;
  current_title: UserTitle | null;
  available_avatars: UserAvatar[];
  available_titles: UserTitle[];
  stats: UserStats;
  level: number;
}

const rarityColors = {
  common: 'bg-gray-100 text-gray-800',
  rare: 'bg-blue-100 text-blue-800',
  epic: 'bg-purple-100 text-purple-800',
  legendary: 'bg-yellow-100 text-yellow-800',
};

const rarityLabels = {
  common: 'コモン',
  rare: 'レア',
  epic: 'エピック',
  legendary: 'レジェンダリー',
};

export default function AvatarsPage() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'avatars' | 'titles' | 'stats'>('avatars');

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace('/auth/login?redirect=/avatars');
      return;
    }
    fetchProfile();
  }, [isAuthenticated, router]);

  const fetchProfile = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/avatars/profile', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setProfile(data);
      } else {
        console.error('プロフィール取得エラー:', response.status, await response.text());
      }
    } catch (error) {
      console.error('プロフィール取得エラー:', error);
    } finally {
      setLoading(false);
    }
  };

  const changeAvatar = async (avatarId: number) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/avatars/avatar/change', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ avatar_id: avatarId }),
      });

      if (response.ok) {
        await fetchProfile(); // プロフィールを再取得
      } else {
        console.error('アバター変更エラー:', response.statusText);
      }
    } catch (error) {
      console.error('アバター変更エラー:', error);
    }
  };

  const changeTitle = async (titleId: number) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/avatars/title/change', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ title_id: titleId }),
      });

      if (response.ok) {
        await fetchProfile(); // プロフィールを再取得
      } else {
        console.error('称号変更エラー:', response.statusText);
      }
    } catch (error) {
      console.error('称号変更エラー:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">読み込み中...</p>
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-600">プロフィールの読み込みに失敗しました</p>
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
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">
                アバター・称号システム
              </h1>
              <p className="mt-2 text-gray-600">学習の成果に応じてアバターや称号を獲得しよう！</p>
            </div>
            <div className="flex items-center space-x-4">
              {/* 現在のアバター表示 */}
              <div className="text-center">
                <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  {profile.current_avatar ? (
                    <img
                      src={profile.current_avatar.avatar.image_url}
                      alt={profile.current_avatar.avatar.name}
                      className="w-12 h-12 rounded-full object-cover"
                    />
                  ) : (
                    <UserCircleIcon className="w-12 h-12 text-indigo-600" />
                  )}
                </div>
                <p className="text-sm font-medium text-gray-900">
                  {profile.current_avatar?.avatar.name || 'デフォルト'}
                </p>
              </div>
              {/* 現在の称号表示 */}
              <div className="text-center">
                <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  {profile.current_title ? (
                    <img
                      src={profile.current_title.title.icon_url}
                      alt={profile.current_title.title.name}
                      className="w-12 h-12 rounded-full object-cover"
                    />
                  ) : (
                    <TrophyIcon className="w-12 h-12 text-yellow-600" />
                  )}
                </div>
                <p className="text-sm font-medium text-gray-900">
                  {profile.current_title?.title.name || '称号なし'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* タブナビゲーション */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('avatars')}
              className={`${
                activeTab === 'avatars'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              <UserCircleIcon className="mr-2 h-5 w-5" />
              アバター
            </button>
            <button
              onClick={() => setActiveTab('titles')}
              className={`${
                activeTab === 'titles'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              <TrophyIcon className="mr-2 h-5 w-5" />
              称号
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`${
                activeTab === 'stats'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              } flex whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium`}
            >
              <SparklesIcon className="mr-2 h-5 w-5" />
              統計
            </button>
          </nav>
        </div>

        {/* アバタータブ */}
        {activeTab === 'avatars' && (
          <div className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {profile.available_avatars.map(userAvatar => (
                <Card key={userAvatar.id} className="relative">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{userAvatar.avatar.name}</CardTitle>
                      <Badge
                        className={
                          rarityColors[userAvatar.avatar.rarity as keyof typeof rarityColors]
                        }
                      >
                        {rarityLabels[userAvatar.avatar.rarity as keyof typeof rarityLabels]}
                      </Badge>
                    </div>
                    <CardDescription>{userAvatar.avatar.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="w-full h-32 bg-gray-100 rounded-lg flex items-center justify-center mb-4">
                      {userAvatar.avatar.image_url ? (
                        <img
                          src={userAvatar.avatar.image_url}
                          alt={userAvatar.avatar.name}
                          className="w-24 h-24 rounded-full object-cover"
                        />
                      ) : (
                        <UserCircleIcon className="w-24 h-24 text-gray-400" />
                      )}
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-gray-600">
                        カテゴリ: {userAvatar.avatar.category}
                      </p>
                      <p className="text-sm text-gray-600">
                        解除日: {new Date(userAvatar.unlocked_at).toLocaleDateString()}
                      </p>
                      <Button
                        onClick={() => changeAvatar(userAvatar.avatar_id)}
                        className="w-full"
                        variant={userAvatar.is_current ? 'secondary' : 'default'}
                        disabled={userAvatar.is_current}
                      >
                        {userAvatar.is_current ? '現在使用中' : '選択する'}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* 称号タブ */}
        {activeTab === 'titles' && (
          <div className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {profile.available_titles.map(userTitle => (
                <Card key={userTitle.id} className="relative">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{userTitle.title.name}</CardTitle>
                      <Badge
                        className={
                          rarityColors[userTitle.title.rarity as keyof typeof rarityColors]
                        }
                      >
                        {rarityLabels[userTitle.title.rarity as keyof typeof rarityLabels]}
                      </Badge>
                    </div>
                    <CardDescription>{userTitle.title.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="w-full h-32 bg-gray-100 rounded-lg flex items-center justify-center mb-4">
                      {userTitle.title.icon_url ? (
                        <img
                          src={userTitle.title.icon_url}
                          alt={userTitle.title.name}
                          className="w-24 h-24 rounded-full object-cover"
                        />
                      ) : (
                        <TrophyIcon className="w-24 h-24 text-gray-400" />
                      )}
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-gray-600">カテゴリ: {userTitle.title.category}</p>
                      <p className="text-sm text-gray-600">
                        獲得日: {new Date(userTitle.unlocked_at).toLocaleDateString()}
                      </p>
                      <Button
                        onClick={() => changeTitle(userTitle.title_id)}
                        className="w-full"
                        variant={userTitle.is_current ? 'secondary' : 'default'}
                        disabled={userTitle.is_current}
                      >
                        {userTitle.is_current ? '現在使用中' : '選択する'}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* 統計タブ */}
        {activeTab === 'stats' && (
          <div className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">レベル</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{profile.level}</div>
                  <p className="text-xs text-gray-500">学習レベル</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">完了クエスト</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{profile.stats.total_quests_completed}</div>
                  <p className="text-xs text-gray-500">総クエスト数</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">連続学習</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{profile.stats.max_streak_days}</div>
                  <p className="text-xs text-gray-500">最大連続日数</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">学習時間</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {Math.round(profile.stats.total_learning_time_minutes / 60)}
                  </div>
                  <p className="text-xs text-gray-500">時間</p>
                </CardContent>
              </Card>
            </div>

            {/* スキルレベル */}
            <Card>
              <CardHeader>
                <CardTitle>スキルレベル</CardTitle>
                <CardDescription>あなたの非認知能力スキルの現在レベル</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>グリット（やり抜く力）</span>
                    <span>{profile.stats.grit_level.toFixed(1)}</span>
                  </div>
                  <Progress value={(profile.stats.grit_level / 5) * 100} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>協調性</span>
                    <span>{profile.stats.collaboration_level.toFixed(1)}</span>
                  </div>
                  <Progress value={(profile.stats.collaboration_level / 5) * 100} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>自己制御</span>
                    <span>{profile.stats.self_regulation_level.toFixed(1)}</span>
                  </div>
                  <Progress
                    value={(profile.stats.self_regulation_level / 5) * 100}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>感情知能</span>
                    <span>{profile.stats.emotional_intelligence_level.toFixed(1)}</span>
                  </div>
                  <Progress
                    value={(profile.stats.emotional_intelligence_level / 5) * 100}
                    className="h-2"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  );
}
