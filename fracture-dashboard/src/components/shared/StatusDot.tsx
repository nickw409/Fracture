const colorMap: Record<string, string> = {
  active: 'bg-emerald-400',
  dead: 'bg-red-400',
  calibrating: 'bg-amber-400',
  connected: 'bg-blue-400',
};

export default function StatusDot({ status }: { status: string }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${colorMap[status] ?? 'bg-gray-500'}`}
    />
  );
}
