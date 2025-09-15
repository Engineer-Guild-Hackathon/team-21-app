import { apiUrl } from '@/lib/api';
('use client');

import { useAuth } from '@/app/contexts/AuthContext';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { ArrowLeft, CheckCircle, Clock, Lightbulb, Save, Star, Trophy, Users } from 'lucide-react';
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
  quest: Quest; // APIはQuestProgressWithQuestを返すため
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
  const [isComposing, setIsComposing] = useState(false);
  const setQuestDataLocal = (data: any) => setQuestData((prev: any) => ({ ...prev, ...data }));

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
      const response = await fetch(apiUrl('/api/quests/'), {
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
      const response = await fetch(apiUrl('/api/quests/my-progress'), {
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

      const response = await fetch(apiUrl(`/api/quests/progress/${progress.id}`), {
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
      const response = await fetch(apiUrl(`/api/quests/progress/${progress.id}`), {
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

    // クエストテンプレート優先（新規行動型クエスト）
    const template = quest.quest_config?.template as
      | 'helping_report'
      | 'achievement_diary'
      | 'ask_for_help'
      | 'streak_habit'
      | 'mini_teacher'
      | 'respect_different_opinion'
      | undefined;

    switch (template) {
      case 'helping_report':
        return renderHelpingReport();
      case 'achievement_diary':
        return renderAchievementDiary();
      case 'ask_for_help':
        return renderAskForHelp();
      case 'streak_habit':
        return renderStreakHabit();
      case 'mini_teacher':
        return renderMiniTeacher();
      case 'respect_different_opinion':
        return renderRespectDifferentOpinion();
      default:
        break;
    }

    // 既存タイプのレンダリング
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

  const renderHelpingReport = () => {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-blue-500" />
              <span>小さな助け合いレポート</span>
            </CardTitle>
            <CardDescription>手助けした内容と相手の反応を記録しましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="help_content">どんな手助けをしましたか？</Label>
              <Textarea
                id="help_content"
                placeholder="例: 落とした教科書を拾って渡した"
                value={questData.help_content || ''}
                onChange={e => setQuestDataLocal({ help_content: e.target.value })}
                className="mt-1"
                rows={3}
              />
            </div>

            <div>
              <Label htmlFor="help_target">手助けした相手</Label>
              <Input
                id="help_target"
                placeholder="例: 友だち / 家族 / 先生"
                value={questData.help_target || ''}
                onChange={e => setQuestDataLocal({ help_target: e.target.value })}
                className="mt-1"
              />
            </div>

            <div>
              <Label htmlFor="reaction">相手の反応</Label>
              <select
                id="reaction"
                className="mt-1 block w-full rounded-md border px-3 py-2 text-sm"
                value={questData.reaction || ''}
                onChange={e => setQuestDataLocal({ reaction: e.target.value })}
              >
                <option value="">選択してください</option>
                <option value="smile">笑顔だった</option>
                <option value="thanks">ありがとうと言ってくれた</option>
                <option value="surprised">驚いていた</option>
                <option value="other">その他</option>
              </select>
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                記録した項目:{' '}
                {
                  [questData.help_content, questData.help_target, questData.reaction].filter(
                    v => v && String(v).trim()
                  ).length
                }{' '}
                / 3
              </div>
              <Button
                onClick={() => {
                  const filled = [
                    questData.help_content,
                    questData.help_target,
                    questData.reaction,
                  ].filter(v => v && String(v).trim()).length;
                  updateProgress({}, (filled / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? '保存中...' : '保存'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderAchievementDiary = () => {
    const items: Array<{ key: string; label: string; placeholder: string }> = [
      { key: 'achieved_1', label: 'できたこと 1', placeholder: '例: 音読を最後までできた' },
      { key: 'achieved_2', label: 'できたこと 2', placeholder: '例: 片付けを自分からやった' },
      { key: 'achieved_3', label: 'できたこと 3', placeholder: '例: 新しい漢字を3つ覚えた' },
    ];
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5 text-yellow-500" />
              <span>できたこと日記</span>
            </CardTitle>
            <CardDescription>今日のできたことを3つ書き、難易度を星で評価しましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {items.map((it, idx) => (
              <div key={it.key} className="space-y-2">
                <Label htmlFor={it.key}>{it.label}</Label>
                <Input
                  id={it.key}
                  placeholder={it.placeholder}
                  value={questData[it.key] || ''}
                  onChange={e => setQuestDataLocal({ [it.key]: e.target.value })}
                  className="mt-1"
                />
                <div className="flex items-center space-x-2">
                  <Label>むずかしさ</Label>
                  <select
                    className="block rounded-md border px-2 py-1 text-sm"
                    value={questData[`difficulty_${idx + 1}`] || ''}
                    onChange={e => setQuestDataLocal({ [`difficulty_${idx + 1}`]: e.target.value })}
                  >
                    <option value="">選択</option>
                    <option value="1">★☆☆☆☆</option>
                    <option value="2">★★☆☆☆</option>
                    <option value="3">★★★☆☆</option>
                    <option value="4">★★★★☆</option>
                    <option value="5">★★★★★</option>
                  </select>
                </div>
              </div>
            ))}

            <div className="flex justify-between items-center pt-2">
              <div className="text-sm text-gray-600">
                入力済み: {items.filter(it => questData[it.key] && questData[it.key].trim()).length}{' '}
                / 3
              </div>
              <Button
                onClick={() => {
                  const count = items.filter(
                    it => questData[it.key] && questData[it.key].trim()
                  ).length;
                  updateProgress({}, (count / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? '保存中...' : '保存'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderAskForHelp = () => {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Users className="w-5 h-5 text-indigo-500" />
              <span>困ったら聞こうチャレンジ</span>
            </CardTitle>
            <CardDescription>質問文を作り、相手に丁寧に聞いてみましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="question_text">質問文</Label>
              <Textarea
                id="question_text"
                placeholder="例: この問題の考え方を教えてください"
                value={questData.question_text || ''}
                onChange={e => setQuestDataLocal({ question_text: e.target.value })}
                className="mt-1"
                rows={3}
              />
            </div>
            <div>
              <Label htmlFor="question_target">誰に聞きましたか？</Label>
              <Input
                id="question_target"
                placeholder="例: 先生 / 友だち / 家族"
                value={questData.question_target || ''}
                onChange={e => setQuestDataLocal({ question_target: e.target.value })}
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="question_result">結果</Label>
              <select
                id="question_result"
                className="mt-1 block w-full rounded-md border px-3 py-2 text-sm"
                value={questData.question_result || ''}
                onChange={e => setQuestDataLocal({ question_result: e.target.value })}
              >
                <option value="">選択してください</option>
                <option value="solved">解決した</option>
                <option value="hint">ヒントをもらえた</option>
                <option value="try_again">後でもう一度聞く</option>
              </select>
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                記録した項目:{' '}
                {
                  [
                    questData.question_text,
                    questData.question_target,
                    questData.question_result,
                  ].filter(v => v && String(v).trim()).length
                }{' '}
                / 3
              </div>
              <Button
                onClick={() => {
                  const filled = [
                    questData.question_text,
                    questData.question_target,
                    questData.question_result,
                  ].filter(v => v && String(v).trim()).length;
                  updateProgress({}, (filled / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? '保存中...' : '保存'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderStreakHabit = () => {
    const habitName = questData.habit_name || '';
    const day1 = !!questData.day1;
    const day2 = !!questData.day2;
    const day3 = !!questData.day3;

    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-emerald-600" />
              <span>途中でやめないリレー（3日連続）</span>
            </CardTitle>
            <CardDescription>短い習慣を3日連続で続けましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="habit_name">続けたい短い習慣</Label>
              <Input
                id="habit_name"
                placeholder="例: 3分片付け / 音読1ページ"
                value={habitName}
                onChange={e => setQuestDataLocal({ habit_name: e.target.value })}
                className="mt-1"
              />
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[1, 2, 3].map(d => (
                <label key={d} className="flex items-center space-x-2 p-3 border rounded">
                  <input
                    type="checkbox"
                    checked={!!questData[`day${d}`]}
                    onChange={e => setQuestDataLocal({ [`day${d}`]: e.target.checked })}
                  />
                  <span>Day {d}</span>
                </label>
              ))}
            </div>

            <div className="flex justify-between items-center pt-2">
              <div className="text-sm text-gray-600">
                連続達成: {[day1, day2, day3].filter(Boolean).length} / 3 日
              </div>
              <Button
                onClick={() => {
                  const cnt = [day1, day2, day3].filter(Boolean).length;
                  updateProgress({}, (cnt / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? '保存中...' : '保存'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderMiniTeacher = () => {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-teal-600" />
              <span>ミニ先生</span>
            </CardTitle>
            <CardDescription>説明→質問→確認の3ステップを記録しましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="teach_topic">教えたこと</Label>
              <Input
                id="teach_topic"
                placeholder="例: 分数の足し算のやり方"
                value={questData.teach_topic || ''}
                onChange={e => setQuestDataLocal({ teach_topic: e.target.value })}
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="teach_question">相手にした質問</Label>
              <Input
                id="teach_question"
                placeholder="例: どこがむずかしかった？"
                value={questData.teach_question || ''}
                onChange={e => setQuestDataLocal({ teach_question: e.target.value })}
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="teach_check">理解の確認</Label>
              <select
                id="teach_check"
                className="mt-1 block w-full rounded-md border px-3 py-2 text-sm"
                value={questData.teach_check || ''}
                onChange={e => setQuestDataLocal({ teach_check: e.target.value })}
              >
                <option value="">選択してください</option>
                <option value="understood">わかったみたい</option>
                <option value="need_more">もう少し説明が必要</option>
              </select>
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                記録した項目:{' '}
                {
                  [questData.teach_topic, questData.teach_question, questData.teach_check].filter(
                    v => v && String(v).trim()
                  ).length
                }{' '}
                / 3
              </div>
              <Button
                onClick={() => {
                  const filled = [
                    questData.teach_topic,
                    questData.teach_question,
                    questData.teach_check,
                  ].filter(v => v && String(v).trim()).length;
                  updateProgress({}, (filled / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? '保存中...' : '保存'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderRespectDifferentOpinion = () => {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Lightbulb className="w-5 h-5 text-orange-500" />
              <span>みんな違ってみんないい</span>
            </CardTitle>
            <CardDescription>自分と違う意見の良いところを見つけて書きましょう</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="other_opinion">相手の意見</Label>
              <Textarea
                id="other_opinion"
                placeholder="相手の意見を簡単に書きましょう"
                value={questData.other_opinion || ''}
                onChange={e => setQuestDataLocal({ other_opinion: e.target.value })}
                className="mt-1"
                rows={3}
              />
            </div>
            <div>
              <Label>良いところを2つ</Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-1">
                <Input
                  placeholder="良いところ 1"
                  value={questData.good_point_1 || ''}
                  onChange={e => setQuestDataLocal({ good_point_1: e.target.value })}
                />
                <Input
                  placeholder="良いところ 2"
                  value={questData.good_point_2 || ''}
                  onChange={e => setQuestDataLocal({ good_point_2: e.target.value })}
                />
              </div>
            </div>

            <div className="flex justify-between items-center pt-4">
              <div className="text-sm text-gray-600">
                入力済み:{' '}
                {
                  [questData.other_opinion, questData.good_point_1, questData.good_point_2].filter(
                    v => v && String(v).trim()
                  ).length
                }{' '}
                / 3
              </div>
              <Button
                onClick={() => {
                  const count = [
                    questData.other_opinion,
                    questData.good_point_1,
                    questData.good_point_2,
                  ].filter(v => v && String(v).trim()).length;
                  updateProgress({}, (count / 3) * 100);
                }}
                disabled={saving}
                variant="outline"
                size="sm"
              >
                {saving ? '保存中...' : '保存'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
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
                onChange={e => setQuestDataLocal({ achievement: e.target.value })}
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
                onChange={e => setQuestDataLocal({ feeling: e.target.value })}
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
                onChange={e => setQuestDataLocal({ gratitude: e.target.value })}
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
                onChange={e => setQuestDataLocal({ prompt: e.target.value })}
                className="mt-1"
              />
            </div>

            <div>
              <Label htmlFor="story">あなたのお話</Label>
              <Textarea
                id="story"
                placeholder="お話を書いてみましょう..."
                value={questData.story || ''}
                onChange={e => setQuestDataLocal({ story: e.target.value })}
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
                onChange={e => setQuestDataLocal({ characters: e.target.value })}
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
                  onCompositionStart={() => setIsComposing(true)}
                  onCompositionEnd={() => setIsComposing(false)}
                  onChange={e => setQuestDataLocal({ pendingHint: e.target.value })}
                  onKeyDown={e => {
                    if (isComposing) return;
                    const value = (questData.pendingHint || '').trim();
                    if (e.key === 'Enter' && value) {
                      const newHints = [...(questData.hints || []), value];
                      setQuestDataLocal({ hints: newHints, pendingHint: '' });
                      updateProgress({ hints: newHints });
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
