'use client';

import { useAuth } from '@/app/contexts/AuthContext';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { ArrowLeft, CheckCircle, Clock, Save, Star, Trophy } from 'lucide-react';
import { useParams, useRouter } from 'next/navigation';
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
  quest_config: any;
}

interface QuestProgress {
  id: number;
  status: string;
  progress_percentage: number;
  current_step: number;
  total_steps: number;
  quest_data: any;
  started_date: string;
  completed_date?: string;
}

export default function QuestDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const questId = params.questId as string;

  const [quest, setQuest] = useState<Quest | null>(null);
  const [progress, setProgress] = useState<QuestProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [questData, setQuestData] = useState<any>({});

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/auth/login');
      return;
    }

    fetchQuest();
    fetchProgress();
  }, [questId, isAuthenticated, router]);

  const fetchQuest = async () => {
    try {
      // クエスト情報は一覧から取得したものを使用
      // 実際の実装では個別のクエスト取得APIを呼び出す
      const response = await fetch('http://localhost:8000/api/quests/', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        const foundQuest = data.quests.find((q: Quest) => q.id === parseInt(questId));
        setQuest(foundQuest);
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
        const foundProgress = data.progress.find(
          (p: QuestProgress) => p.quest.id === parseInt(questId)
        );
        setProgress(foundProgress);
        if (foundProgress && foundProgress.quest_data) {
          setQuestData(foundProgress.quest_data);
        }
      }
    } catch (error) {
      console.error('進捗取得エラー:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateProgress = async (newData: any, newProgress?: number) => {
    if (!progress) return;

    setSaving(true);
    try {
      const updateData = {
        quest_data: { ...questData, ...newData },
        progress_percentage: newProgress || progress.progress_percentage,
      };

      const response = await fetch(`http://localhost:8000/api/quests/progress/${progress.id}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });

      if (response.ok) {
        setQuestData(updateData.quest_data);
        await fetchProgress(); // 進捗を再取得
      }
    } catch (error) {
      console.error('進捗更新エラー:', error);
    } finally {
      setSaving(false);
    }
  };

  const completeQuest = async () => {
    if (!progress) return;

    setSaving(true);
    try {
      const response = await fetch(`http://localhost:8000/api/quests/progress/${progress.id}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          status: 'completed',
          progress_percentage: 100,
          completed_date: new Date().toISOString(),
        }),
      });

      if (response.ok) {
        await fetchProgress();
        // 完了メッセージを表示
        alert('クエスト完了！おめでとうございます！');
        router.push('/quests');
      }
    } catch (error) {
      console.error('クエスト完了エラー:', error);
    } finally {
      setSaving(false);
    }
  };

  const renderQuestContent = () => {
    if (!quest) return null;

    switch (quest.quest_type) {
      case 'daily_log':
        return renderDailyLogQuest();
      case 'plant_care':
        return renderPlantCareQuest();
      case 'story_creation':
        return renderStoryCreationQuest();
      case 'collaboration':
        return renderCollaborationQuest();
      default:
        return <div>このクエストタイプはまだ実装されていません。</div>;
    }
  };

  const renderDailyLogQuest = () => {
    const today = new Date().toLocaleDateString('ja-JP');

    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span>今日の冒険ログ - {today}</span>
            </CardTitle>
            <CardDescription>今日頑張ったことや成長を感じたことを記録しましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="achievement">今日頑張ったこと</Label>
              <Textarea
                id="achievement"
                placeholder="例: 難しい算数の問題を最後まで解いた"
                value={questData.achievement || ''}
                onChange={e => updateProgress({ achievement: e.target.value })}
                className="mt-1"
                rows={3}
              />
            </div>

            <div>
              <Label htmlFor="feeling">その時の気持ち</Label>
              <Textarea
                id="feeling"
                placeholder="例: 最初は難しかったけど、最後まで頑張れて嬉しかった"
                value={questData.feeling || ''}
                onChange={e => updateProgress({ feeling: e.target.value })}
                className="mt-1"
                rows={3}
              />
            </div>

            <div>
              <Label htmlFor="gratitude">感謝したいこと</Label>
              <Textarea
                id="gratitude"
                placeholder="例: 先生が優しく教えてくれたこと"
                value={questData.gratitude || ''}
                onChange={e => updateProgress({ gratitude: e.target.value })}
                className="mt-1"
                rows={2}
              />
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                記録した項目:{' '}
                {
                  [questData.achievement, questData.feeling, questData.gratitude].filter(
                    item => item && item.trim()
                  ).length
                }{' '}
                / 3
              </div>
              <Button
                onClick={() => {
                  const completedItems = [
                    questData.achievement,
                    questData.feeling,
                    questData.gratitude,
                  ].filter(item => item && item.trim()).length;
                  updateProgress({}, (completedItems / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? (
                  '保存中...'
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-1" />
                    保存
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderPlantCareQuest = () => {
    const growthStages = [
      { stage: 1, name: '種', description: '種を植えました', progress: 0 },
      { stage: 2, name: '芽', description: '小さな芽が出ました', progress: 25 },
      { stage: 3, name: '葉', description: '緑の葉が育ちました', progress: 50 },
      { stage: 4, name: 'つぼみ', description: '花のつぼみができました', progress: 75 },
      { stage: 5, name: '花', description: '美しい花が咲きました', progress: 100 },
    ];

    const currentStage = questData.currentStage || 1;
    const daysCared = questData.daysCared || 0;

    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5 text-green-500" />
              <span>魔法の種を育てよう</span>
            </CardTitle>
            <CardDescription>毎日お世話をして、美しい花を咲かせましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 植物の成長状況 */}
            <div className="text-center">
              <div className="text-6xl mb-4">
                {currentStage === 1 && '🌱'}
                {currentStage === 2 && '🌿'}
                {currentStage === 3 && '🌳'}
                {currentStage === 4 && '🌺'}
                {currentStage === 5 && '🌸'}
              </div>
              <h3 className="text-xl font-semibold mb-2">{growthStages[currentStage - 1]?.name}</h3>
              <p className="text-gray-600 mb-4">{growthStages[currentStage - 1]?.description}</p>
              <Progress value={growthStages[currentStage - 1]?.progress || 0} className="h-3" />
            </div>

            {/* お世話記録 */}
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{daysCared}</div>
                <div className="text-sm text-gray-600">お世話した日数</div>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{currentStage}/5</div>
                <div className="text-sm text-gray-600">成長段階</div>
              </div>
            </div>

            {/* 今日のお世話 */}
            <div className="space-y-4">
              <h4 className="font-semibold">今日のお世話</h4>

              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    id="water"
                    checked={questData.wateredToday || false}
                    onChange={e => updateProgress({ wateredToday: e.target.checked })}
                    className="w-4 h-4 text-blue-600"
                  />
                  <Label htmlFor="water">水をあげる</Label>
                </div>

                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    id="sunlight"
                    checked={questData.sunlightToday || false}
                    onChange={e => updateProgress({ sunlightToday: e.target.checked })}
                    className="w-4 h-4 text-blue-600"
                  />
                  <Label htmlFor="sunlight">日光に当てる</Label>
                </div>

                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    id="talk"
                    checked={questData.talkedToday || false}
                    onChange={e => updateProgress({ talkedToday: e.target.checked })}
                    className="w-4 h-4 text-blue-600"
                  />
                  <Label htmlFor="talk">優しく話しかける</Label>
                </div>
              </div>

              <div className="pt-4">
                <Button
                  onClick={() => {
                    const careCount = [
                      questData.wateredToday,
                      questData.sunlightToday,
                      questData.talkedToday,
                    ].filter(Boolean).length;
                    const newDaysCared = daysCared + (careCount >= 2 ? 1 : 0);
                    const newStage = Math.min(5, Math.floor(newDaysCared / 3) + 1);

                    updateProgress(
                      {
                        daysCared: newDaysCared,
                        currentStage: newStage,
                        wateredToday: false,
                        sunlightToday: false,
                        talkedToday: false,
                      },
                      (newStage / 5) * 100
                    );
                  }}
                  disabled={saving}
                  className="w-full"
                >
                  {saving ? '保存中...' : 'お世話を記録する'}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderStoryCreationQuest = () => {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5 text-purple-500" />
              <span>お話の森</span>
            </CardTitle>
            <CardDescription>創造力を働かせて、素敵なお話を作りましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="prompt">お話のテーマ</Label>
              <Input
                id="prompt"
                placeholder="例: 魔法の森で出会った不思議な友達"
                value={questData.prompt || ''}
                onChange={e => updateProgress({ prompt: e.target.value })}
                className="mt-1"
              />
            </div>

            <div>
              <Label htmlFor="story">あなたのお話</Label>
              <Textarea
                id="story"
                placeholder="お話を書いてみましょう..."
                value={questData.story || ''}
                onChange={e => updateProgress({ story: e.target.value })}
                className="mt-1"
                rows={8}
              />
            </div>

            <div>
              <Label htmlFor="characters">登場人物</Label>
              <Input
                id="characters"
                placeholder="例: 主人公の少年、魔法の森の妖精"
                value={questData.characters || ''}
                onChange={e => updateProgress({ characters: e.target.value })}
                className="mt-1"
              />
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                文字数: {questData.story?.length || 0}文字
              </div>
              <Button
                onClick={() => {
                  const storyLength = questData.story?.length || 0;
                  const progress = Math.min(100, (storyLength / 500) * 100);
                  updateProgress({}, progress);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? (
                  '保存中...'
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-1" />
                    保存
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderCollaborationQuest = () => {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5 text-blue-500" />
              <span>協力！謎解きチャットルーム</span>
            </CardTitle>
            <CardDescription>みんなで協力して謎を解きましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h4 className="font-semibold mb-2">現在の謎</h4>
              <p className="text-gray-700">
                魔法の森で迷子になった妖精を助けるために、3つの鍵を集める必要があります。
                それぞれの鍵は異なる場所に隠されており、ヒントを手がかりに探す必要があります。
              </p>
            </div>

            <div className="space-y-3">
              <h4 className="font-semibold">発見したヒント</h4>
              <div className="space-y-2">
                {questData.hints?.map((hint: string, index: number) => (
                  <div key={index} className="p-3 bg-gray-50 rounded border-l-4 border-blue-500">
                    <span className="font-medium">ヒント {index + 1}:</span> {hint}
                  </div>
                )) || (
                  <div className="text-gray-500 text-center py-4">
                    まだヒントは見つかっていません
                  </div>
                )}
              </div>
            </div>

            <div>
              <Label htmlFor="newHint">新しいヒントを追加</Label>
              <div className="flex space-x-2 mt-1">
                <Input
                  id="newHint"
                  placeholder="ヒントを入力..."
                  onKeyPress={e => {
                    if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                      const newHints = [...(questData.hints || []), e.currentTarget.value.trim()];
                      updateProgress({ hints: newHints });
                      e.currentTarget.value = '';
                    }
                  }}
                />
                <Button
                  onClick={() => {
                    const input = document.getElementById('newHint') as HTMLInputElement;
                    if (input.value.trim()) {
                      const newHints = [...(questData.hints || []), input.value.trim()];
                      updateProgress({ hints: newHints });
                      input.value = '';
                    }
                  }}
                  size="sm"
                >
                  追加
                </Button>
              </div>
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                発見したヒント: {questData.hints?.length || 0}個
              </div>
              <Button
                onClick={() => {
                  const hintCount = questData.hints?.length || 0;
                  const progress = Math.min(100, (hintCount / 5) * 100);
                  updateProgress({}, progress);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? (
                  '保存中...'
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-1" />
                    保存
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
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

  if (!quest || !progress) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">クエストが見つかりません</h1>
          <Button onClick={() => router.push('/quests')}>クエスト一覧に戻る</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* ヘッダー */}
      <div className="mb-8">
        <Button onClick={() => router.push('/quests')} variant="outline" className="mb-4">
          <ArrowLeft className="w-4 h-4 mr-2" />
          クエスト一覧に戻る
        </Button>

        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">{quest.title}</h1>
            <p className="text-gray-600">{quest.description}</p>
          </div>
          <div className="text-right">
            <Badge className="mb-2">{quest.target_skill}</Badge>
            <div className="flex items-center space-x-4 text-sm text-gray-600">
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
          </div>
        </div>
      </div>

      {/* 進捗表示 */}
      <Card className="mb-8">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">進捗</span>
            <span className="text-sm text-gray-600">
              {progress.progress_percentage.toFixed(0)}%
            </span>
          </div>
          <Progress value={progress.progress_percentage} className="h-3" />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>ステップ {progress.current_step}</span>
            <span>総ステップ {progress.total_steps}</span>
          </div>
        </CardContent>
      </Card>

      {/* クエストコンテンツ */}
      {renderQuestContent()}

      {/* 完了ボタン */}
      {progress.progress_percentage >= 100 && progress.status !== 'completed' && (
        <div className="mt-8 text-center">
          <Button
            onClick={completeQuest}
            disabled={saving}
            size="lg"
            className="bg-green-600 hover:bg-green-700"
          >
            {saving ? '完了処理中...' : 'クエスト完了！'}
          </Button>
        </div>
      )}
    </div>
  );
}
