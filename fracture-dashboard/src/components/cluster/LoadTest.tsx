import { useLoadTest } from '../../hooks/useLoadTest';
import { Play, Square } from 'lucide-react';

export default function LoadTest() {
  const { stats, start, stop } = useLoadTest();

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
          Load Test
        </h3>
        {stats.running ? (
          <button
            onClick={stop}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-red-400/10 text-red-400 rounded-md hover:bg-red-400/20 transition-colors duration-150"
          >
            <Square size={12} /> Stop
          </button>
        ) : (
          <button
            onClick={() => start(3000, 64)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-emerald-400/10 text-emerald-400 rounded-md hover:bg-emerald-400/20 transition-colors duration-150"
          >
            <Play size={12} /> Start
          </button>
        )}
      </div>

      <div className="grid grid-cols-4 gap-3">
        <div>
          <div className="text-xs text-gray-500">Sent</div>
          <div className="font-mono text-sm">{stats.requestsSent}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Completed</div>
          <div className="font-mono text-sm text-emerald-400">{stats.requestsCompleted}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Failed</div>
          <div className="font-mono text-sm text-red-400">{stats.requestsFailed}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Tokens</div>
          <div className="font-mono text-sm">{stats.totalTokens}</div>
        </div>
      </div>
    </div>
  );
}
