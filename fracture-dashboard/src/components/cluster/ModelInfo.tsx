import type { ModelInfo as ModelInfoType } from '../../api/types';

export default function ModelInfoCard({ model }: { model: ModelInfoType }) {
  const items = [
    { label: 'Model', value: model.name },
    { label: 'Parameters', value: model.parameters },
    { label: 'Layers', value: model.layers.toString() },
    { label: 'Context', value: model.context_length.toLocaleString() },
    { label: 'Dtype', value: model.dtype },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
        Model
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
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
