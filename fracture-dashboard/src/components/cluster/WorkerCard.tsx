import type { WorkerInfo } from '../../api/types';
import StatusDot from '../shared/StatusDot';
import ProgressBar from '../shared/ProgressBar';

const borderColor: Record<string, string> = {
  active: 'border-emerald-400/30',
  dead: 'border-red-400/30',
  calibrating: 'border-amber-400/30',
};

export default function WorkerCard({ worker }: { worker: WorkerInfo }) {
  const vramUsage = worker.vram_total_mb > 0 ? worker.vram_used_mb / worker.vram_total_mb : 0;

  return (
    <div
      className={`bg-gray-900 border rounded-lg p-4 min-w-[200px] ${
        borderColor[worker.status] ?? 'border-gray-700'
      }`}
    >
      <div className="flex items-center gap-2 mb-3">
        <StatusDot status={worker.status} />
        <span className="text-sm font-medium truncate">{worker.gpu}</span>
      </div>

      <div className="space-y-2">
        <div>
          <div className="text-xs text-gray-400 mb-1">
            Layers {worker.layers[0]}&ndash;{worker.layers[1]}
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>VRAM</span>
            <span className="font-mono">
              {(worker.vram_used_mb / 1024).toFixed(1)} / {(worker.vram_total_mb / 1024).toFixed(1)} GB
            </span>
          </div>
          <ProgressBar value={vramUsage} colorClass="bg-violet-400" />
        </div>

        {worker.calibration_ms_per_layer > 0 && (
          <div className="text-xs text-gray-500">
            <span className="font-mono">{worker.calibration_ms_per_layer.toFixed(2)}</span> ms/layer
          </div>
        )}
      </div>
    </div>
  );
}
