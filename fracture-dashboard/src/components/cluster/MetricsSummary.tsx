import type { MetricsEvent } from '../../api/types';
import Sparkline from '../shared/Sparkline';
import { formatThroughput, formatDuration, formatPercent } from '../../lib/formatters';

interface Props {
  metrics: MetricsEvent[];
  latest: MetricsEvent | null;
}

export default function MetricsSummary({ metrics, latest }: Props) {
  const cards = [
    {
      label: 'Throughput',
      value: latest ? `${formatThroughput(latest.throughput_tokens_per_sec)} tok/s` : '--',
      data: metrics.map((m) => m.throughput_tokens_per_sec),
      color: '#34d399',
    },
    {
      label: 'Active Requests',
      value: latest ? latest.active_requests.toString() : '--',
      data: metrics.map((m) => m.active_requests),
      color: '#60a5fa',
    },
    {
      label: 'TTFT',
      value: latest ? formatDuration(latest.avg_time_to_first_token_ms) : '--',
      data: metrics.map((m) => m.avg_time_to_first_token_ms),
      color: '#fbbf24',
    },
    {
      label: 'Cache Utilization',
      value: latest ? formatPercent(latest.kv_cache_utilization) : '--',
      data: metrics.map((m) => m.kv_cache_utilization * 100),
      color: '#a78bfa',
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map(({ label, value, data, color }) => (
        <div key={label} className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">{label}</div>
          <div className="font-mono text-xl mb-2">{value}</div>
          {data.length > 1 && <Sparkline data={data} color={color} />}
        </div>
      ))}
    </div>
  );
}
