'use client';

interface NonCogSkillCardProps {
  name: string;
  score: number;
  icon: any;
  color: string;
  description: string;
  recentActivity: string;
}

export default function NonCogSkillCard({
  name,
  score,
  icon: Icon,
  color,
  description,
  recentActivity,
}: NonCogSkillCardProps) {
  return (
    <div className={`rounded-lg p-6 ${color}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <Icon className="h-6 w-6 mr-2" />
          <h3 className="font-semibold">{name}</h3>
        </div>
        <div className="text-2xl font-bold">{Math.round(score * 100)}</div>
      </div>
      <div className="mb-4">
        <div className="w-full bg-white rounded-full h-2">
          <div
            className="bg-current h-2 rounded-full transition-all duration-500"
            style={{ width: `${score * 100}%` }}
          />
        </div>
      </div>
      <p className="text-sm mb-2">{description}</p>
      <p className="text-sm opacity-75">{recentActivity}</p>
    </div>
  );
}
