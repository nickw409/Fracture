import type { GenerationStats as Stats } from '../../hooks/useInference';
import { formatDuration } from '../../lib/formatters';

export default function GenerationStats({ stats }: { stats: Stats }) {
  const items = [
    {
      label: 'TTFT',
      value: stats.timeToFirstTokenMs != null ? formatDuration(stats.timeToFirstTokenMs) : '--',
    },
    {
      label: 'Speed',
      value: stats.tokensPerSecond != null ? `${stats.tokensPerSecond.toFixed(1)} tok/s` : '--',
    },
    {
      label: 'Tokens',
      value: stats.completionTokens > 0 ? stats.completionTokens.toString() : '--',
    },
    {
      label: 'Finish',
      value: stats.finishReason ?? '--',
    },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
        Generation
      </h3>
      <div className="grid grid-cols-2 gap-3">
        {items.map(({ label, value }) => (
          <div key={label}>
            <div className="text-xs text-gray-500">{label}</div>
            <div className="font-mono text-sm">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
