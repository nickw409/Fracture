interface Props {
  value: number; // 0-1
  className?: string;
  colorClass?: string;
}

export default function ProgressBar({ value, className = '', colorClass }: Props) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  const color =
    colorClass ?? (pct < 60 ? 'bg-emerald-400' : pct < 85 ? 'bg-amber-400' : 'bg-red-400');

  return (
    <div className={`h-2 bg-gray-800 rounded-full overflow-hidden ${className}`}>
      <div
        className={`h-full rounded-full transition-all duration-300 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
