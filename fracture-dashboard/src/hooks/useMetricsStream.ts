import { useState, useEffect, useRef } from 'react';
import type { MetricsEvent } from '../api/types';
import { mockMetrics } from '../api/mock';
import { METRICS_BUFFER_SIZE } from '../lib/constants';

const MOCK = import.meta.env.VITE_MOCK === 'true';

export function useMetricsStream(bufferSize = METRICS_BUFFER_SIZE) {
  const [metrics, setMetrics] = useState<MetricsEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const mockRef = useRef<ReturnType<typeof setInterval>>(undefined);

  useEffect(() => {
    if (MOCK) {
      setConnected(true);
      mockRef.current = setInterval(() => {
        setMetrics((prev) => {
          const next = [...prev, mockMetrics()];
          return next.length > bufferSize ? next.slice(-bufferSize) : next;
        });
      }, 1000);
      return () => clearInterval(mockRef.current);
    }

    // Connect directly to the backend — SSE doesn't work reliably through Vite's proxy.
    const base = import.meta.env.VITE_API_URL ?? `${window.location.protocol}//${window.location.hostname}:8080`;
    const es = new EventSource(`${base}/v1/metrics/stream`);

    es.onopen = () => setConnected(true);

    es.onmessage = (event) => {
      const data: MetricsEvent = JSON.parse(event.data);
      setMetrics((prev) => {
        const next = [...prev, data];
        return next.length > bufferSize ? next.slice(-bufferSize) : next;
      });
    };

    es.onerror = () => setConnected(false);

    return () => es.close();
  }, [bufferSize]);

  return { metrics, connected, latest: metrics[metrics.length - 1] ?? null };
}
