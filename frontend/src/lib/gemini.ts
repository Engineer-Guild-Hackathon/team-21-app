import { GoogleGenerativeAI } from '@google/generative-ai';

// Gemini API の設定
const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY || '');

// 学習に特化したプロンプト
const LEARNING_SYSTEM_PROMPT = `あなたは学習支援に特化したAIアシスタントです。以下の特徴を持っています：

🎯 役割：
- 学習者の質問に丁寧に答える
- 宿題や課題の解決をサポートする
- 勉強方法やコツを教える
- 学習への動機づけを行う

📚 対応科目：
- 数学：計算、図形、関数、統計など
- 理科：物理、化学、生物、地学
- 国語：読解、作文、文法、漢字
- 社会：歴史、地理、公民
- 英語：文法、単語、会話、読解

💡 回答の特徴：
- 分かりやすく、段階的に説明する
- 具体例を使って理解を深める
- 学習者のレベルに合わせた説明
- 励ましの言葉も含める
- 必要に応じて図や表の提案もする

🎨 非認知能力の向上：
- グリット（やり抜く力）を育てる
- 協調性を促す
- 自己制御力を高める
- 感情知能を向上させる

常に学習者を励まし、成長をサポートする姿勢で回答してください。`;

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export class GeminiChatService {
  private model: any;
  private chatSession: any;

  constructor() {
    this.model = genAI.getGenerativeModel({
      model: 'gemini-1.5-flash',
      systemInstruction: LEARNING_SYSTEM_PROMPT,
    });
    this.chatSession = this.model.startChat({
      history: [],
    });
  }

  async sendMessage(message: string): Promise<string> {
    try {
      if (!process.env.NEXT_PUBLIC_GEMINI_API_KEY) {
        throw new Error('Gemini API key is not configured');
      }

      const result = await this.chatSession.sendMessage(message);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Gemini API Error:', error);

      // API キーが設定されていない場合のフォールバック
      if (error instanceof Error && error.message.includes('API key')) {
        return this.getFallbackResponse(message);
      }

      throw new Error('AIチャットでエラーが発生しました。もう一度お試しください。');
    }
  }

  private getFallbackResponse(userMessage: string): string {
    // API キーが設定されていない場合のモック応答
    const responses = [
      'とても良い質問ですね！',
      'その問題について一緒に考えてみましょう。',
      '素晴らしいアイデアです！',
      'その疑問は多くの人が持つものです。',
      'とても興味深い内容ですね。',
    ];

    const randomResponse = responses[Math.floor(Math.random() * responses.length)];

    return (
      `${randomResponse}\n\nあなたの質問「${userMessage}」について、AIアシスタントが詳しくお答えします。\n\n` +
      '現在はデモモードです。実際のAI機能を使用するには、Gemini API キーを設定してください。\n\n' +
      '📚 学習サポート機能：\n' +
      '• 数学・理科・国語・社会・英語の質問対応\n' +
      '• 宿題の手伝い\n' +
      '• 勉強方法のアドバイス\n' +
      '• 非認知能力の向上サポート'
    );
  }

  // チャット履歴をリセット
  resetChat() {
    this.chatSession = this.model.startChat({
      history: [],
    });
  }

  // 特定の科目に特化したプロンプトを生成
  static generateSubjectPrompt(subject: string, topic: string): string {
    const prompts = {
      math: `数学の「${topic}」について質問があります。基礎から応用まで、段階的に教えてください。`,
      science: `理科の「${topic}」について質問があります。実験や観察のポイントも含めて教えてください。`,
      japanese: `国語の「${topic}」について質問があります。読解のコツや表現方法を教えてください。`,
      social: `社会の「${topic}」について質問があります。歴史的背景や地理的特徴も含めて教えてください。`,
      english: `英語の「${topic}」について質問があります。文法や表現方法を教えてください。`,
    };

    return prompts[subject as keyof typeof prompts] || `「${topic}」について質問があります。`;
  }
}

// シングルトンインスタンス
export const geminiChatService = new GeminiChatService();
