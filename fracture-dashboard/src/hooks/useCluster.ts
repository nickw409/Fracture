import { useState, useEffect } from 'react';
import type { ClusterResponse } from '../api/types';
import { api } from '../api/client';
import { mockCluster } from '../api/mock';
import { CLUSTER_POLL_MS } from '../lib/constants';

const MOCK = import.meta.env.VITE_MOCK === 'true';

export function useCluster(intervalMs = CLUSTER_POLL_MS) {
  const [data, setData] = useState<ClusterResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const result = MOCK ? mockCluster() : await api.cluster();
        if (active) {
          setData(result);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setLoading(false);
        }
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { data, error, loading };
}
