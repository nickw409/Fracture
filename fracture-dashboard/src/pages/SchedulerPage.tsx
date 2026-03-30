import { useScheduler } from '../hooks/useScheduler';
import QueueOverview from '../components/scheduler/QueueOverview';
import CachePool from '../components/scheduler/CachePool';
import SequenceTable from '../components/scheduler/SequenceTable';
import Skeleton from '../components/shared/Skeleton';

export default function SchedulerPage() {
  const { data, error, loading } = useScheduler();

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-3 gap-4">
          {[...Array(3)].map((_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
        <Skeleton className="h-24" />
        <Skeleton className="h-48" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-400/10 border border-red-400/30 rounded-lg p-4">
        <div className="text-red-400 text-sm">
          Failed to load scheduler data: {error?.message ?? 'unknown error'}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Scheduler</h1>
      <QueueOverview data={data} />
      <CachePool cache={data.kv_cache} />

      <div>
        <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
          Active Sequences
        </h2>
        <SequenceTable sequences={data.sequences} />
      </div>
    </div>
  );
}
