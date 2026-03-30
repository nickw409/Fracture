import type { KvCacheInfo } from '../../api/types';
import ProgressBar from '../shared/ProgressBar';
import { formatPercent } from '../../lib/formatters';

export default function CachePool({ cache }: { cache: KvCacheInfo }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
        KV Cache Block Pool
      </h3>

      <ProgressBar value={cache.utilization} className="h-4 mb-3" />

      <div className="flex justify-between text-sm">
        <span className="text-gray-400">
          <span className="font-mono">{cache.allocated_blocks.toLocaleString()}</span> /{' '}
          <span className="font-mono">{cache.total_blocks.toLocaleString()}</span> blocks
          <span className="text-gray-500 ml-1">({formatPercent(cache.utilization)})</span>
        </span>
        <span className="text-gray-500 text-xs">
          {cache.block_size} tokens/block
        </span>
      </div>
    </div>
  );
}
