import type {
  ClusterResponse,
  MetricsEvent,
  SchedulerResponse,
  RequestsResponse,
} from './types';

export function mockCluster(): ClusterResponse {
  return {
    mode: 'standalone',
    num_workers: 1,
    workers: [
      {
        id: 0,
        role: 'standalone',
        address: 'local',
        gpu: 'NVIDIA RTX 4090',
        vram_total_mb: 24576,
        vram_used_mb: 15360,
        layers: [0, 31],
        status: 'active',
        last_heartbeat_ms: 0,
        calibration_ms_per_layer: 1.2,
      },
    ],
    scheduling_mode: 'auto',
    model: {
      name: 'llama-3-8b',
      parameters: '8B',
      layers: 32,
      context_length: 4096,
      dtype: 'FP16',
    },
  };
}

let metricsTime = 0;
export function mockMetrics(): MetricsEvent {
  metricsTime++;
  return {
    timestamp: new Date().toISOString(),
    throughput_tokens_per_sec: 45 + Math.sin(metricsTime * 0.2) * 15 + Math.random() * 5,
    active_requests: Math.floor(Math.random() * 4),
    avg_time_to_first_token_ms: 120 + Math.random() * 40,
    avg_inter_token_latency_ms: 18 + Math.random() * 8,
    kv_cache_utilization: 0.35 + Math.sin(metricsTime * 0.1) * 0.15,
    worker_heartbeats: [0],
  };
}

export function mockScheduler(): SchedulerResponse {
  return {
    active_sequences: 2,
    max_sequences: 64,
    decode_queue: 2,
    prefill_queue: 0,
    prefill_chunk_size: 512,
    kv_cache: {
      block_size: 16,
      total_blocks: 2048,
      allocated_blocks: 847,
      free_blocks: 1201,
      utilization: 0.413,
    },
    sequences: [
      {
        id: 'seq-0',
        state: 'decoding',
        tokens_generated: 42,
        max_tokens: 256,
        prefill_tokens: 0,
        cache_blocks_held: 12,
        started_at: new Date(Date.now() - 3000).toISOString(),
      },
      {
        id: 'seq-1',
        state: 'prefilling',
        tokens_generated: 0,
        max_tokens: 128,
        prefill_tokens: 87,
        cache_blocks_held: 6,
        started_at: new Date(Date.now() - 500).toISOString(),
      },
    ],
  };
}

export function mockRequests(): RequestsResponse {
  return {
    requests: [
      {
        id: 'req-001',
        type: 'chat',
        status: 'completed',
        prompt_tokens: 128,
        completion_tokens: 96,
        total_tokens: 224,
        time_to_first_token_ms: 145,
        total_duration_ms: 2100,
        tokens_per_second: 45.7,
        finish_reason: 'stop',
        temperature: 0.7,
        created_at: new Date(Date.now() - 10000).toISOString(),
      },
      {
        id: 'req-002',
        type: 'completion',
        status: 'completed',
        prompt_tokens: 64,
        completion_tokens: 256,
        total_tokens: 320,
        time_to_first_token_ms: 95,
        total_duration_ms: 5600,
        tokens_per_second: 45.7,
        finish_reason: 'length',
        temperature: 0.0,
        created_at: new Date(Date.now() - 30000).toISOString(),
      },
    ],
    total: 2,
    page: 1,
    per_page: 50,
  };
}
