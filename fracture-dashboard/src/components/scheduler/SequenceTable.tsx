import { useState } from 'react';
import type { SequenceInfo } from '../../api/types';

const stateBadge: Record<string, string> = {
  prefilling: 'bg-blue-400/10 text-blue-400',
  decoding: 'bg-emerald-400/10 text-emerald-400',
  completed: 'bg-gray-700 text-gray-400',
};

type SortKey = 'id' | 'state' | 'tokens_generated' | 'max_tokens';

export default function SequenceTable({ sequences }: { sequences: SequenceInfo[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = [...sequences].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const cmp = typeof av === 'number' ? (av as number) - (bv as number) : String(av).localeCompare(String(bv));
    return sortAsc ? cmp : -cmp;
  });

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const headers: { key: SortKey; label: string }[] = [
    { key: 'id', label: 'ID' },
    { key: 'state', label: 'State' },
    { key: 'tokens_generated', label: 'Tokens' },
    { key: 'max_tokens', label: 'Max' },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 text-xs uppercase">
            {headers.map(({ key, label }) => (
              <th
                key={key}
                onClick={() => toggleSort(key)}
                className="text-left px-4 py-2 cursor-pointer hover:text-gray-200 transition-colors duration-150 select-none"
              >
                {label} {sortKey === key && (sortAsc ? '↑' : '↓')}
              </th>
            ))}
            <th className="text-left px-4 py-2">Prefill</th>
            <th className="text-left px-4 py-2">Blocks</th>
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr>
              <td colSpan={6} className="px-4 py-6 text-center text-gray-500">
                No active sequences
              </td>
            </tr>
          )}
          {sorted.map((seq) => (
            <tr key={seq.id} className="border-t border-gray-800">
              <td className="px-4 py-2 font-mono text-xs">{seq.id}</td>
              <td className="px-4 py-2">
                <span className={`inline-block px-2 py-0.5 rounded text-xs ${stateBadge[seq.state] ?? ''}`}>
                  {seq.state}
                </span>
              </td>
              <td className="px-4 py-2 font-mono">
                {seq.tokens_generated} / {seq.max_tokens}
              </td>
              <td className="px-4 py-2 font-mono">{seq.max_tokens}</td>
              <td className="px-4 py-2 font-mono">{seq.prefill_tokens}</td>
              <td className="px-4 py-2 font-mono">{seq.cache_blocks_held}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
