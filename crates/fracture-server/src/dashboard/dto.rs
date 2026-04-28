use serde::Serialize;

// ── Cluster ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ClusterResponse {
    pub mode: &'static str,
    pub num_workers: usize,
    pub workers: Vec<WorkerInfo>,
    pub scheduling_mode: &'static str,
    pub model: ModelInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    pub id: usize,
    pub role: &'static str,
    pub address: String,
    pub gpu: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub layers: [usize; 2],
    pub status: &'static str,
    pub last_heartbeat_ms: u64,
    pub calibration_ms_per_layer: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: String,
    pub layers: usize,
    pub context_length: usize,
    pub dtype: String,
}

// ── Scheduler ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct SchedulerResponse {
    pub active_sequences: usize,
    pub max_sequences: usize,
    pub decode_queue: usize,
    pub prefill_queue: usize,
    pub prefill_chunk_size: usize,
    pub kv_cache: KvCacheInfo,
    pub sequences: Vec<SequenceInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KvCacheInfo {
    pub block_size: usize,
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub utilization: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SequenceInfo {
    pub id: String,
    pub state: &'static str,
    pub tokens_generated: usize,
    pub max_tokens: usize,
    pub prefill_tokens: usize,
    pub cache_blocks_held: usize,
    pub started_at: String,
}

// ── Metrics stream ───────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct MetricsEvent {
    pub timestamp: String,
    pub throughput_tokens_per_sec: f64,
    pub active_requests: usize,
    pub avg_time_to_first_token_ms: f64,
    pub avg_inter_token_latency_ms: f64,
    pub kv_cache_utilization: f64,
    pub worker_heartbeats: Vec<u64>,
}

// ── Request log ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct RequestsResponse {
    pub requests: Vec<RequestRecord>,
    pub total: usize,
    pub page: usize,
    pub per_page: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    pub id: String,
    #[serde(rename = "type")]
    pub request_type: &'static str,
    pub status: &'static str,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub time_to_first_token_ms: f64,
    pub total_duration_ms: f64,
    pub tokens_per_second: f64,
    pub finish_reason: &'static str,
    pub temperature: f32,
    pub created_at: String,
}

// ── Scheduler snapshot (internal, for oneshot channel) ───

#[derive(Debug, Clone)]
pub struct SchedulerSnapshot {
    pub active_sequences: usize,
    pub max_sequences: usize,
    pub decode_count: usize,
    pub prefill_queue_count: usize,
    pub prefill_chunk_size: usize,
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub sequences: Vec<SequenceSnapshotEntry>,
}

#[derive(Debug, Clone)]
pub struct SequenceSnapshotEntry {
    pub seq_id: u64,
    pub state: &'static str,
    pub tokens_generated: usize,
    pub max_tokens: usize,
    pub remaining_prefill: usize,
}

impl SchedulerSnapshot {
    pub fn to_response(&self) -> SchedulerResponse {
        let allocated = self.total_blocks.saturating_sub(self.free_blocks);
        let utilization = if self.total_blocks > 0 {
            allocated as f64 / self.total_blocks as f64
        } else {
            0.0
        };

        SchedulerResponse {
            active_sequences: self.active_sequences,
            max_sequences: self.max_sequences,
            decode_queue: self.decode_count,
            prefill_queue: self.prefill_queue_count,
            prefill_chunk_size: self.prefill_chunk_size,
            kv_cache: KvCacheInfo {
                block_size: self.block_size,
                total_blocks: self.total_blocks,
                allocated_blocks: allocated,
                free_blocks: self.free_blocks,
                utilization,
            },
            sequences: self
                .sequences
                .iter()
                .map(|s| SequenceInfo {
                    id: format!("seq-{}", s.seq_id),
                    state: s.state,
                    tokens_generated: s.tokens_generated,
                    max_tokens: s.max_tokens,
                    prefill_tokens: s.remaining_prefill,
                    cache_blocks_held: 0, // not tracked at scheduler level
                    started_at: String::new(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_response_serializes() {
        let resp = ClusterResponse {
            mode: "standalone",
            num_workers: 1,
            workers: vec![WorkerInfo {
                id: 0,
                role: "standalone",
                address: "local".to_string(),
                gpu: "RTX 4090".to_string(),
                vram_total_mb: 24576,
                vram_used_mb: 15000,
                layers: [0, 31],
                status: "active",
                last_heartbeat_ms: 0,
                calibration_ms_per_layer: 1.2,
            }],
            scheduling_mode: "auto",
            model: ModelInfo {
                name: "llama-3-8b".to_string(),
                parameters: "8B".to_string(),
                layers: 32,
                context_length: 4096,
                dtype: "FP16".to_string(),
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["mode"], "standalone");
        assert_eq!(json["num_workers"], 1);
        assert_eq!(json["workers"][0]["gpu"], "RTX 4090");
        assert_eq!(json["workers"][0]["layers"][0], 0);
        assert_eq!(json["workers"][0]["layers"][1], 31);
        assert_eq!(json["model"]["name"], "llama-3-8b");
    }

    #[test]
    fn test_scheduler_snapshot_to_response() {
        let snap = SchedulerSnapshot {
            active_sequences: 3,
            max_sequences: 64,
            decode_count: 2,
            prefill_queue_count: 1,
            prefill_chunk_size: 512,
            block_size: 16,
            total_blocks: 2048,
            free_blocks: 1200,
            sequences: vec![
                SequenceSnapshotEntry {
                    seq_id: 0,
                    state: "decoding",
                    tokens_generated: 42,
                    max_tokens: 256,
                    remaining_prefill: 0,
                },
                SequenceSnapshotEntry {
                    seq_id: 1,
                    state: "prefilling",
                    tokens_generated: 0,
                    max_tokens: 128,
                    remaining_prefill: 50,
                },
            ],
        };

        let resp = snap.to_response();
        assert_eq!(resp.active_sequences, 3);
        assert_eq!(resp.max_sequences, 64);
        assert_eq!(resp.decode_queue, 2);
        assert_eq!(resp.prefill_queue, 1);
        assert_eq!(resp.kv_cache.total_blocks, 2048);
        assert_eq!(resp.kv_cache.free_blocks, 1200);
        assert_eq!(resp.kv_cache.allocated_blocks, 848);
        let expected_util = 848.0 / 2048.0;
        assert!((resp.kv_cache.utilization - expected_util).abs() < 0.001);
        assert_eq!(resp.sequences.len(), 2);
        assert_eq!(resp.sequences[0].id, "seq-0");
        assert_eq!(resp.sequences[0].state, "decoding");
        assert_eq!(resp.sequences[1].state, "prefilling");
    }

    #[test]
    fn test_scheduler_snapshot_empty_blocks_zero_utilization() {
        let snap = SchedulerSnapshot {
            active_sequences: 0,
            max_sequences: 0,
            decode_count: 0,
            prefill_queue_count: 0,
            prefill_chunk_size: 0,
            block_size: 16,
            total_blocks: 0,
            free_blocks: 0,
            sequences: vec![],
        };
        let resp = snap.to_response();
        assert_eq!(resp.kv_cache.utilization, 0.0);
    }

    #[test]
    fn test_request_record_serializes_with_type_rename() {
        let rec = RequestRecord {
            id: "req-1".to_string(),
            request_type: "chat",
            status: "completed",
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            time_to_first_token_ms: 100.0,
            total_duration_ms: 500.0,
            tokens_per_second: 40.0,
            finish_reason: "stop",
            temperature: 0.7,
            created_at: "2026-03-29T00:00:00Z".to_string(),
        };
        let json = serde_json::to_value(&rec).unwrap();
        // Field is renamed from request_type to type.
        assert_eq!(json["type"], "chat");
        assert!(json.get("request_type").is_none());
        assert_eq!(json["finish_reason"], "stop");
    }

    #[test]
    fn test_metrics_event_serializes() {
        let ev = MetricsEvent {
            timestamp: "2026-03-29T00:00:00.000Z".to_string(),
            throughput_tokens_per_sec: 45.0,
            active_requests: 2,
            avg_time_to_first_token_ms: 120.0,
            avg_inter_token_latency_ms: 22.0,
            kv_cache_utilization: 0.41,
            worker_heartbeats: vec![0, 150],
        };
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["throughput_tokens_per_sec"], 45.0);
        assert_eq!(json["worker_heartbeats"][1], 150);
    }
}
