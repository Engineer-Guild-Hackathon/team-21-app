'use client';

import { CheckIcon } from '@heroicons/react/24/outline';
import { useState } from 'react';

interface TermsOfServiceProps {
  onAccept: (accepted: boolean) => void;
  accepted: boolean;
}

export default function TermsOfService({ onAccept, accepted }: TermsOfServiceProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onAccept(e.target.checked);
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <input
            type="checkbox"
            id="terms-agreement"
            checked={accepted}
            onChange={handleCheckboxChange}
            className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
        </div>
        <div className="flex-1 min-w-0">
          <label
            htmlFor="terms-agreement"
            className="text-sm font-medium text-gray-900 cursor-pointer"
          >
            <span className="flex items-center">
              <CheckIcon className="h-4 w-4 text-green-500 mr-2" />
              Non-Cog 利用同意書に同意します
            </span>
          </label>
          <p className="text-xs text-gray-500 mt-1">アカウント作成には利用規約への同意が必要です</p>
        </div>
      </div>

      {/* 同意書の詳細表示 */}
      <div className="mt-4">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-sm text-blue-600 hover:text-blue-800 font-medium"
        >
          {isExpanded ? '同意書を閉じる' : '同意書の内容を確認する'}
        </button>

        {isExpanded && (
          <div className="mt-4 max-h-96 overflow-y-auto border border-gray-200 rounded-lg p-4 bg-gray-50">
            <div className="prose prose-sm max-w-none">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Non-Cog 利用同意書</h3>

              <p className="text-sm text-gray-700 mb-4">
                本サービス（以下「本アプリ」といいます。）は、Non-Cog（以下「当社」といいます。）が提供する、非認知能力を育むための学習アプリケーションです。本アプリをご利用いただく前に、以下の利用規約にご同意いただく必要があります。
              </p>

              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gray-900">第1条（適用）</h4>
                  <ol className="list-decimal list-inside text-sm text-gray-700 mt-2 space-y-1">
                    <li>
                      本規約は、本アプリの提供条件及び本アプリの利用に関する当社とユーザーとの間の権利義務関係を定めることを目的とし、ユーザーと当社との間の本アプリの利用に関わる一切の関係に適用されます。
                    </li>
                    <li>
                      ユーザーは、本アプリのアカウント作成を完了した時点で、本規約の全ての記載内容に同意したものとみなされます。
                    </li>
                  </ol>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-900">第2条（プライバシーポリシー）</h4>
                  <ol className="list-decimal list-inside text-sm text-gray-700 mt-2 space-y-1">
                    <li>
                      当社は、ユーザーの個人情報（氏名、生年月日、連絡先、所属クラス情報など）を、当社のプライバシーポリシーに基づき、適切に管理・保護します。
                    </li>
                    <li>
                      当社は、本アプリを通じて取得したユーザーの学習データ（クエストのクリア状況、チャットログ、成果物など）を、本アプリの改善、非認知能力の育成に関する研究、及び先生へのフィードバック提供のためにのみ利用します。
                    </li>
                    <li>
                      当社は、法令で定められた場合を除き、ユーザーの個人情報をユーザーの同意なく第三者に開示または提供することはありません。
                    </li>
                  </ol>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-900">第3条（行動規範）</h4>
                  <p className="text-sm text-gray-700 mt-2">
                    ユーザーは、本アプリの利用において、以下の行為を行ってはならないものとします。
                  </p>
                  <ul className="list-disc list-inside text-sm text-gray-700 mt-2 space-y-1 ml-4">
                    <li>他のユーザー、先生、または当社の名誉や信用を毀損する行為</li>
                    <li>他のユーザーへの誹謗中傷、嫌がらせ、または差別的な言動</li>
                    <li>個人情報やプライバシーを侵害する行為</li>
                    <li>不適切なコンテンツ（性的、暴力的など）を投稿または送信する行為</li>
                    <li>違法行為、または違法行為を助長する行為</li>
                    <li>本アプリの運営を妨害する行為</li>
                    <li>当社または第三者の著作権、商標権、その他の知的財産権を侵害する行為</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-900">第4条（データ管理とセキュリティ）</h4>
                  <ol className="list-decimal list-inside text-sm text-gray-700 mt-2 space-y-1">
                    <li>
                      当社は、ユーザーの個人情報および学習データを、不正アクセス、紛失、破壊、改ざん、および漏洩から保護するため、適切なセキュリティ対策を講じます。
                    </li>
                    <li>
                      本アプリを通じて送受信される重要なデータは、
                      <strong className="text-blue-600">
                        業界標準の暗号化技術を用いて安全に管理
                      </strong>
                      します。
                    </li>
                    <li>
                      当社は、システムの障害や緊急事態に備え、データのバックアップを定期的に取得します。
                    </li>
                  </ol>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-900">第5条（免責事項）</h4>
                  <ol className="list-decimal list-inside text-sm text-gray-700 mt-2 space-y-1">
                    <li>
                      当社は、本アプリの利用によりユーザーに生じた損害について、当社の故意または重過失による場合を除き、一切の責任を負わないものとします。
                    </li>
                    <li>
                      当社は、本アプリの提供の停止、中止、終了によりユーザーに生じた損害について、一切の責任を負わないものとします。
                    </li>
                  </ol>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  <strong>重要：</strong>{' '}
                  アカウント作成を完了することで、上記の利用規約に同意したものとみなされます。
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 同意状況の表示 */}
      {accepted && (
        <div className="mt-4 flex items-center text-green-600">
          <CheckIcon className="h-5 w-5 mr-2" />
          <span className="text-sm font-medium">同意書に同意済み</span>
        </div>
      )}
    </div>
  );
}
