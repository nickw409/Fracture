interface Params {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  seed: string;
}

interface Props {
  params: Params;
  onChange: (params: Params) => void;
}

export default function ParamControls({ params, onChange }: Props) {
  const set = <K extends keyof Params>(key: K, value: Params[K]) =>
    onChange({ ...params, [key]: value });

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Parameters</h3>

      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Temperature</span>
          <span className="font-mono">{params.temperature.toFixed(1)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={params.temperature}
          onChange={(e) => set('temperature', parseFloat(e.target.value))}
          className="w-full accent-blue-400"
        />
      </div>

      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Top-P</span>
          <span className="font-mono">{params.topP.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={params.topP}
          onChange={(e) => set('topP', parseFloat(e.target.value))}
          className="w-full accent-blue-400"
        />
      </div>

      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Top-K</span>
          <span className="font-mono">{params.topK}</span>
        </div>
        <input
          type="number"
          min="0"
          value={params.topK}
          onChange={(e) => set('topK', parseInt(e.target.value) || 0)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-sm font-mono text-gray-100 focus:outline-none focus:border-gray-600"
        />
      </div>

      <div>
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Max Tokens</span>
          <span className="font-mono">{params.maxTokens}</span>
        </div>
        <input
          type="number"
          min="1"
          max="4096"
          value={params.maxTokens}
          onChange={(e) => set('maxTokens', parseInt(e.target.value) || 256)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-sm font-mono text-gray-100 focus:outline-none focus:border-gray-600"
        />
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-1">Seed</div>
        <input
          type="text"
          placeholder="Random"
          value={params.seed}
          onChange={(e) => set('seed', e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-sm font-mono text-gray-100 placeholder-gray-600 focus:outline-none focus:border-gray-600"
        />
      </div>
    </div>
  );
}
