"use client";

import { useState } from "react";

// メタデータはサーバーコンポーネントでのみ動作するため、別ファイルに移動する必要があります
// export const metadata: Metadata = {
//   title: "感情コントロール - 非認知能力学習プラットフォーム",
//   description: "ストレス管理と感情コントロールを学ぶ",
// };

interface EmotionLog {
  emotion: string;
  intensity: number;
  trigger: string;
  timestamp: Date;
}

export default function EmotionControlPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [emotionLogs, setEmotionLogs] = useState<EmotionLog[]>([]);
  const [currentEmotion, setCurrentEmotion] = useState<string>("");
  const [emotionIntensity, setEmotionIntensity] = useState<number>(5);
  const [trigger, setTrigger] = useState<string>("");

  const handleEmotionLog = () => {
    const newLog: EmotionLog = {
      emotion: currentEmotion,
      intensity: emotionIntensity,
      trigger: trigger,
      timestamp: new Date(),
    };
    setEmotionLogs([...emotionLogs, newLog]);
    setCurrentEmotion("");
    setEmotionIntensity(5);
    setTrigger("");
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 md:p-24">
      <div className="w-full max-w-4xl">
        <h1 className="text-4xl font-bold mb-8 gradient-text">
          感情コントロール入門
        </h1>

        {/* ステップ1: 感情認識 */}
        <div className="card p-8 mb-8">
          <h2 className="text-2xl font-semibold mb-4">
            ステップ1: 今の感情を認識する
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                現在の感情
              </label>
              <select
                value={currentEmotion}
                onChange={(e) => setCurrentEmotion(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">選択してください</option>
                <option value="joy">喜び</option>
                <option value="anger">怒り</option>
                <option value="sadness">悲しみ</option>
                <option value="fear">不安</option>
                <option value="stress">ストレス</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                感情の強さ（1-10）
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={emotionIntensity}
                onChange={(e) => setEmotionIntensity(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-500">
                <span>弱</span>
                <span>強</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                きっかけ
              </label>
              <textarea
                value={trigger}
                onChange={(e) => setTrigger(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                rows={3}
                placeholder="この感情を感じるきっかけとなった出来事を書いてください"
              />
            </div>

            <button
              onClick={handleEmotionLog}
              className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
              disabled={!currentEmotion || !trigger}
            >
              記録する
            </button>
          </div>
        </div>

        {/* ステップ2: 感情ログの振り返り */}
        {emotionLogs.length > 0 && (
          <div className="card p-8 mb-8">
            <h2 className="text-2xl font-semibold mb-4">
              ステップ2: 感情の振り返り
            </h2>
            <div className="space-y-4">
              {emotionLogs.map((log, index) => (
                <div
                  key={index}
                  className="border-l-4 border-blue-500 pl-4 py-2"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium">
                        感情: {log.emotion} (強さ: {log.intensity})
                      </p>
                      <p className="text-gray-600">{log.trigger}</p>
                    </div>
                    <p className="text-sm text-gray-500">
                      {log.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ステップ3: 対処法の提案 */}
        {emotionLogs.length >= 3 && (
          <div className="card p-8">
            <h2 className="text-2xl font-semibold mb-4">
              ステップ3: 感情コントロールの方法
            </h2>
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-md">
                <h3 className="font-medium mb-2">呼吸法</h3>
                <p>
                  1. 楽な姿勢で座ります
                  <br />
                  2. 目を閉じるか、一点を見つめます
                  <br />
                  3. ゆっくりと4秒かけて息を吸います
                  <br />
                  4. 4秒間息を止めます
                  <br />
                  5. 6秒かけてゆっくりと息を吐きます
                  <br />
                  6. これを5回繰り返します
                </p>
              </div>

              <div className="bg-purple-50 p-4 rounded-md">
                <h3 className="font-medium mb-2">マインドフルネス</h3>
                <p>
                  1. 現在の瞬間に意識を向けます
                  <br />
                  2. 周りの音、匂い、感覚に注目します
                  <br />
                  3. 思考や感情を判断せずに観察します
                  <br />
                  4. 呼吸に意識を戻します
                </p>
              </div>

              <div className="bg-green-50 p-4 rounded-md">
                <h3 className="font-medium mb-2">認知の再構成</h3>
                <p>
                  1. 感情を引き起こす考えを特定します
                  <br />
                  2. その考えは事実に基づいていますか？
                  <br />
                  3. 別の見方はありませんか？
                  <br />
                  4. より建設的な考え方を見つけます
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
