import type {
  ClusterResponse,
  HealthResponse,
  RequestsResponse,
  SchedulerResponse,
} from './types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => fetchJSON<HealthResponse>('/health'),
  cluster: () => fetchJSON<ClusterResponse>('/v1/cluster'),
  scheduler: () => fetchJSON<SchedulerResponse>('/v1/scheduler'),
  requests: (page = 1, perPage = 50) =>
    fetchJSON<RequestsResponse>(`/v1/requests?page=${page}&per_page=${perPage}`),
};
