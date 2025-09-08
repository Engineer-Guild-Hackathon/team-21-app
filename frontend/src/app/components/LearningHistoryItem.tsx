'use client';

interface LearningHistoryItemProps {
  date: string;
  activity: string;
  duration: string;
  emotions: string[];
  achievement: string;
}

export default function LearningHistoryItem({
  date,
  activity,
  duration,
  emotions,
  achievement,
}: LearningHistoryItemProps) {
  const getEmotionColor = (emotion: string) => {
    const colors: { [key: string]: string } = {
      集中: 'bg-blue-100 text-blue-800',
      喜び: 'bg-yellow-100 text-yellow-800',
      協力: 'bg-green-100 text-green-800',
      興奮: 'bg-purple-100 text-purple-800',
      フラストレーション: 'bg-red-100 text-red-800',
      達成感: 'bg-indigo-100 text-indigo-800',
    };
    return colors[emotion] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-medium text-gray-900">{activity}</h3>
        <span className="text-sm text-gray-500">{date}</span>
      </div>
      <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500">
        <div className="flex items-center">
          <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          {duration}
        </div>
        <div className="flex flex-wrap gap-2">
          {emotions.map(emotion => (
            <span
              key={emotion}
              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getEmotionColor(
                emotion
              )}`}
            >
              {emotion}
            </span>
          ))}
        </div>
        <div className="flex items-center">
          <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          {achievement}
        </div>
      </div>
    </div>
  );
}
