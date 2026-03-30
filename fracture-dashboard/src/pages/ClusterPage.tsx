import { useCluster } from '../hooks/useCluster';
import { useMetricsStream } from '../hooks/useMetricsStream';
import PipelineView from '../components/cluster/PipelineView';
import ModelInfoCard from '../components/cluster/ModelInfo';
import MetricsSummary from '../components/cluster/MetricsSummary';
import Skeleton from '../components/shared/Skeleton';

export default function ClusterPage() {
  const { data: cluster, error, loading } = useCluster();
  const { metrics, latest } = useMetricsStream();

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !cluster) {
    return (
      <div className="bg-red-400/10 border border-red-400/30 rounded-lg p-4">
        <div className="text-red-400 text-sm">
          Failed to load cluster data: {error?.message ?? 'unknown error'}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Cluster Overview</h1>

      <div>
        <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
          Pipeline
        </h2>
        <PipelineView workers={cluster.workers} />
      </div>

      <MetricsSummary metrics={metrics} latest={latest} />

      <ModelInfoCard model={cluster.model} />
    </div>
  );
}
