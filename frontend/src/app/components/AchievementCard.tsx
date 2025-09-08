'use client';

interface AchievementCardProps {
  title: string;
  date: string;
  description: string;
  type: 'achievement' | 'leadership' | 'streak';
}

export default function AchievementCard({ title, date, description, type }: AchievementCardProps) {
  return (
    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
      <div>
        <h3 className="font-medium text-gray-900">{title}</h3>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
      <div className="flex flex-col items-end">
        <div className="text-sm text-gray-500">{date}</div>
        <div className="text-xs mt-1">
          {type === 'achievement' && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
              達成
            </span>
          )}
          {type === 'leadership' && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
              リーダーシップ
            </span>
          )}
          {type === 'streak' && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
              継続
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
