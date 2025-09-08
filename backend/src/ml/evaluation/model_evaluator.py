from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

from ..dialogue.bert_dialogue import DialogueSystem
from ..emotion_analysis.emotion_analyzer import EmotionAnalyzer
from ..reinforcement.dqn_agent import DQNAgent

class ModelEvaluator:
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 評価用のメトリクス
        self.metrics = {
            "emotion_analysis": {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            },
            "dialogue": {
                "response_appropriateness": [],
                "emotion_consistency": [],
                "context_relevance": []
            },
            "reinforcement": {
                "average_reward": [],
                "action_distribution": {},
                "state_coverage": []
            }
        }
    
    def evaluate_emotion_analyzer(
        self,
        model: EmotionAnalyzer,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """感情分析モデルの評価"""
        y_true = []
        y_pred = []
        
        for sample in test_data:
            # 実際の感情ラベル
            true_emotion = sample["emotion"]
            
            # モデルの予測
            if "text" in sample:
                result = model.analyze_text(sample["text"])
            elif "image" in sample:
                result = model.analyze_image(sample["image"])
            else:
                continue
            
            pred_emotion = result["emotion"]
            
            y_true.append(true_emotion)
            y_pred.append(pred_emotion)
        
        # 評価指標の計算
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # 混同行列の可視化
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Emotion Analysis')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_dir, 'emotion_confusion_matrix.png'))
        plt.close()
        
        return {
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
    
    def evaluate_dialogue_system(
        self,
        model: DialogueSystem,
        test_conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """対話システムの評価"""
        metrics = {
            "response_appropriateness": [],
            "emotion_consistency": [],
            "context_relevance": []
        }
        
        for conv in test_conversations:
            user_message = conv["user_message"]
            expected_response = conv["expected_response"]
            context = conv.get("context", {})
            
            # モデルの応答を生成
            response = model.generate_response(
                user_message=user_message,
                emotion=context.get("emotion", "neutral"),
                action=context.get("action", {"type": "general"})
            )
            
            # 応答の適切性を評価（簡易的なスコアリング）
            appropriateness = self._calculate_response_appropriateness(
                response["text"],
                expected_response
            )
            
            # 感情の一貫性を評価
            emotion_consistency = self._calculate_emotion_consistency(
                response["emotion"],
                context.get("emotion", "neutral")
            )
            
            # コンテキストの関連性を評価
            context_relevance = self._calculate_context_relevance(
                response["text"],
                context
            )
            
            metrics["response_appropriateness"].append(appropriateness)
            metrics["emotion_consistency"].append(emotion_consistency)
            metrics["context_relevance"].append(context_relevance)
        
        # 結果の集計
        results = {
            "average_appropriateness": np.mean(metrics["response_appropriateness"]),
            "average_emotion_consistency": np.mean(metrics["emotion_consistency"]),
            "average_context_relevance": np.mean(metrics["context_relevance"]),
            "total_conversations": len(test_conversations)
        }
        
        # 結果の可視化
        self._plot_dialogue_metrics(metrics)
        
        return results
    
    def evaluate_dqn_agent(
        self,
        agent: DQNAgent,
        test_episodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """強化学習エージェントの評価"""
        episode_rewards = []
        action_counts = {action["type"]: 0 for action in agent.actions}
        state_visits = set()
        
        for episode in test_episodes:
            total_reward = 0
            state = episode["initial_state"]
            
            for step in range(len(episode["transitions"])):
                # 状態の記録
                state_key = self._get_state_key(state)
                state_visits.add(state_key)
                
                # 行動の選択
                action = agent.select_action(state)
                action_counts[action["type"]] += 1
                
                # 報酬の計算
                next_state = episode["transitions"][step]["next_state"]
                reward = agent._calculate_reward(state, next_state, action)
                total_reward += reward
                
                state = next_state
            
            episode_rewards.append(total_reward)
        
        # 結果の集計
        results = {
            "average_reward": np.mean(episode_rewards),
            "reward_std": np.std(episode_rewards),
            "action_distribution": action_counts,
            "state_coverage": len(state_visits),
            "total_episodes": len(test_episodes)
        }
        
        # 結果の可視化
        self._plot_dqn_metrics(episode_rewards, action_counts)
        
        return results
    
    def _calculate_response_appropriateness(
        self,
        generated_response: str,
        expected_response: str
    ) -> float:
        """応答の適切性を計算（0-1のスコア）"""
        # ここでより高度な評価メトリクスを実装可能
        # 現在は簡易的な実装
        common_words = set(generated_response.split()) & set(expected_response.split())
        total_words = set(generated_response.split()) | set(expected_response.split())
        return len(common_words) / len(total_words) if total_words else 0
    
    def _calculate_emotion_consistency(
        self,
        predicted_emotion: str,
        context_emotion: str
    ) -> float:
        """感情の一貫性を計算（0-1のスコア）"""
        # 感情の類似度を定義
        emotion_similarity = {
            ("joy", "concentration"): 0.8,
            ("frustration", "neutral"): 0.5,
            # 他の感情ペアも定義可能
        }
        
        if predicted_emotion == context_emotion:
            return 1.0
        
        # 類似度を確認
        emotion_pair = tuple(sorted([predicted_emotion, context_emotion]))
        return emotion_similarity.get(emotion_pair, 0.0)
    
    def _calculate_context_relevance(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> float:
        """コンテキストとの関連性を計算（0-1のスコア）"""
        # コンテキストに応じたキーワードを定義
        context_keywords = {
            "learning": ["学習", "理解", "問題", "解決"],
            "challenge": ["挑戦", "目標", "達成", "頑張る"],
            "support": ["サポート", "助言", "アドバイス", "手助け"]
        }
        
        if "type" not in context:
            return 0.5  # デフォルトスコア
        
        # コンテキストタイプに関連するキーワードの出現をチェック
        keywords = context_keywords.get(context["type"], [])
        if not keywords:
            return 0.5
        
        # キーワードの出現率を計算
        matched_keywords = sum(1 for word in keywords if word in response)
        return matched_keywords / len(keywords)
    
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """状態を一意のキーに変換"""
        return f"{state['emotion']}_{state['intensity']:.2f}_{state['context']}"
    
    def _plot_dialogue_metrics(self, metrics: Dict[str, List[float]]):
        """対話システムの評価指標を可視化"""
        plt.figure(figsize=(12, 6))
        
        # メトリクスの箱ひげ図
        plt.boxplot([
            metrics["response_appropriateness"],
            metrics["emotion_consistency"],
            metrics["context_relevance"]
        ], labels=[
            "Response\nAppropriateness",
            "Emotion\nConsistency",
            "Context\nRelevance"
        ])
        
        plt.title("Dialogue System Evaluation Metrics")
        plt.ylabel("Score")
        plt.savefig(os.path.join(self.output_dir, 'dialogue_metrics.png'))
        plt.close()
    
    def _plot_dqn_metrics(
        self,
        episode_rewards: List[float],
        action_counts: Dict[str, int]
    ):
        """DQNエージェントの評価指標を可視化"""
        # 報酬の推移
        plt.figure(figsize=(12, 4))
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(os.path.join(self.output_dir, 'dqn_rewards.png'))
        plt.close()
        
        # 行動分布
        plt.figure(figsize=(8, 6))
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        plt.bar(actions, counts)
        plt.title("Action Distribution")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dqn_actions.png'))
        plt.close()
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """評価結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 結果をJSONとして保存
        results_path = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"評価結果を保存しました: {results_path}")

def evaluate_all_models():
    """全モデルの評価を実行"""
    evaluator = ModelEvaluator()
    
    try:
        # モデルのインスタンス化
        emotion_analyzer = EmotionAnalyzer()
        dialogue_system = DialogueSystem()
        dqn_agent = DQNAgent()
        
        # テストデータの読み込み（実際のデータに置き換える）
        test_data = []  # 感情分析用テストデータ
        test_conversations = []  # 対話システム用テストデータ
        test_episodes = []  # 強化学習用テストデータ
        
        # 各モデルの評価
        print("感情分析モデルの評価中...")
        emotion_results = evaluator.evaluate_emotion_analyzer(
            emotion_analyzer,
            test_data
        )
        
        print("対話システムの評価中...")
        dialogue_results = evaluator.evaluate_dialogue_system(
            dialogue_system,
            test_conversations
        )
        
        print("強化学習エージェントの評価中...")
        dqn_results = evaluator.evaluate_dqn_agent(
            dqn_agent,
            test_episodes
        )
        
        # 全体の結果を集約
        all_results = {
            "emotion_analysis": emotion_results,
            "dialogue_system": dialogue_results,
            "dqn_agent": dqn_results,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # 結果の保存
        evaluator.save_evaluation_results(all_results)
        print("評価が完了しました")
        
    except Exception as e:
        print(f"評価中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    evaluate_all_models()
