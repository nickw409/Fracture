import { useState, useEffect } from 'react';
import type { SchedulerResponse } from '../api/types';
import { api } from '../api/client';
import { mockScheduler } from '../api/mock';
import { SCHEDULER_POLL_MS } from '../lib/constants';

const MOCK = import.meta.env.VITE_MOCK === 'true';

export function useScheduler(intervalMs = SCHEDULER_POLL_MS) {
  const [data, setData] = useState<SchedulerResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const result = MOCK ? mockScheduler() : await api.scheduler();
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
