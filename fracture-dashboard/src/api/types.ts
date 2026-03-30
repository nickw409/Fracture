// ── Cluster ──────────────────────────────────────────────

export interface ClusterResponse {
  mode: 'standalone' | 'distributed';
  num_workers: number;
  workers: WorkerInfo[];
  scheduling_mode: 'auto' | 'equal_split' | 'manual';
  model: ModelInfo;
}

export interface WorkerInfo {
  id: number;
  role: 'head' | 'middle' | 'tail' | 'standalone';
  address: string;
  gpu: string;
  vram_total_mb: number;
  vram_used_mb: number;
  layers: [number, number];
  status: 'active' | 'dead' | 'calibrating';
  last_heartbeat_ms: number;
  calibration_ms_per_layer: number;
}

export interface ModelInfo {
  name: string;
  parameters: string;
  layers: number;
  context_length: number;
  dtype: string;
}

// ── Scheduler ────────────────────────────────────────────

export interface SchedulerResponse {
  active_sequences: number;
  max_sequences: number;
  decode_queue: number;
  prefill_queue: number;
  prefill_chunk_size: number;
  kv_cache: KvCacheInfo;
  sequences: SequenceInfo[];
}

export interface KvCacheInfo {
  block_size: number;
  total_blocks: number;
  allocated_blocks: number;
  free_blocks: number;
  utilization: number;
}

export interface SequenceInfo {
  id: string;
  state: 'prefilling' | 'decoding' | 'completed';
  tokens_generated: number;
  max_tokens: number;
  prefill_tokens: number;
  cache_blocks_held: number;
  started_at: string;
}

// ── Metrics stream ───────────────────────────────────────

export interface MetricsEvent {
  timestamp: string;
  throughput_tokens_per_sec: number;
  active_requests: number;
  avg_time_to_first_token_ms: number;
  avg_inter_token_latency_ms: number;
  kv_cache_utilization: number;
  worker_heartbeats: number[];
}

// ── Request log ──────────────────────────────────────────

export interface RequestsResponse {
  requests: RequestRecord[];
  total: number;
  page: number;
  per_page: number;
}

export interface RequestRecord {
  id: string;
  type: 'chat' | 'completion';
  status: 'completed' | 'cancelled' | 'error';
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  time_to_first_token_ms: number;
  total_duration_ms: number;
  tokens_per_second: number;
  finish_reason: 'stop' | 'length' | 'cancelled' | 'error';
  temperature: number;
  created_at: string;
}

// ── Chat completions ─────────────────────────────────────

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
  stream?: boolean;
  stop?: string[];
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
    };
    finish_reason: 'stop' | 'length' | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ── Health ────────────────────────────────────────────────

export interface HealthResponse {
  status: 'ok' | 'ready';
}
