import type { SchedulerResponse } from '../../api/types';

export default function QueueOverview({ data }: { data: SchedulerResponse }) {
  const cards = [
    {
      label: 'Active Sequences',
      value: `${data.active_sequences} / ${data.max_sequences}`,
      accent: 'text-emerald-400',
    },
    {
      label: 'Decode Queue',
      value: data.decode_queue.toString(),
      accent: 'text-blue-400',
    },
    {
      label: 'Prefill Queue',
      value: data.prefill_queue.toString(),
      accent: 'text-amber-400',
    },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {cards.map(({ label, value, accent }) => (
        <div key={label} className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">{label}</div>
          <div className={`font-mono text-2xl ${accent}`}>{value}</div>
        </div>
      ))}
    </div>
  );
}
