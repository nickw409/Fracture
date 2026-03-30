import type { WorkerInfo } from '../../api/types';
import WorkerCard from './WorkerCard';
import { ChevronRight } from 'lucide-react';

export default function PipelineView({ workers }: { workers: WorkerInfo[] }) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-2">
      {workers.map((worker, i) => (
        <div key={worker.id} className="flex items-center gap-2">
          <WorkerCard worker={worker} />
          {i < workers.length - 1 && (
            <ChevronRight className="text-gray-600 flex-shrink-0" size={20} />
          )}
        </div>
      ))}
    </div>
  );
}
