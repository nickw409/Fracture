import { useState, useEffect } from 'react';
import type { RequestsResponse } from '../api/types';
import { api } from '../api/client';
import { mockRequests } from '../api/mock';
import { REQUESTS_POLL_MS } from '../lib/constants';

const MOCK = import.meta.env.VITE_MOCK === 'true';

export function useRequests(intervalMs = REQUESTS_POLL_MS) {
  const [data, setData] = useState<RequestsResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const result = MOCK ? mockRequests() : await api.requests();
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
