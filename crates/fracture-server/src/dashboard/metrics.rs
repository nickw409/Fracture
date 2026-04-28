use super::dto::{MetricsEvent, RequestRecord};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

/// Aggregates request-level stats into time-series metrics for the SSE stream.
pub struct MetricsCollector {
    inner: Mutex<Inner>,
}

struct Inner {
    /// Completed tokens with timestamp, for throughput calculation.
    token_window: VecDeque<(Instant, usize)>,
    /// EMA of time-to-first-token (ms).
    ema_ttft_ms: f64,
    /// EMA of inter-token latency (ms).
    ema_itl_ms: f64,
    /// Current active request count (incremented on start, decremented on finish).
    active_requests: usize,
    /// Smoothing factor for EMA (0..1). Higher = more weight to recent values.
    alpha: f64,
    /// Window duration for throughput (seconds).
    window_secs: f64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                token_window: VecDeque::new(),
                ema_ttft_ms: 0.0,
                ema_itl_ms: 0.0,
                active_requests: 0,
                alpha: 0.3,
                window_secs: 10.0,
            }),
        }
    }

    /// Call when a new request starts.
    pub fn request_started(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.active_requests += 1;
    }

    /// Call when a request completes. Records throughput and latency stats.
    pub fn record_completion(&self, record: &RequestRecord) {
        let mut inner = self.inner.lock().unwrap();
        inner.active_requests = inner.active_requests.saturating_sub(1);

        // Record tokens for throughput window.
        inner
            .token_window
            .push_back((Instant::now(), record.completion_tokens));

        // Update EMA for TTFT.
        if record.time_to_first_token_ms > 0.0 {
            if inner.ema_ttft_ms == 0.0 {
                inner.ema_ttft_ms = record.time_to_first_token_ms;
            } else {
                inner.ema_ttft_ms = inner.alpha * record.time_to_first_token_ms
                    + (1.0 - inner.alpha) * inner.ema_ttft_ms;
            }
        }

        // Update EMA for inter-token latency.
        if record.completion_tokens > 1 && record.total_duration_ms > record.time_to_first_token_ms
        {
            let generation_ms = record.total_duration_ms - record.time_to_first_token_ms;
            let itl = generation_ms / (record.completion_tokens - 1) as f64;
            if inner.ema_itl_ms == 0.0 {
                inner.ema_itl_ms = itl;
            } else {
                inner.ema_itl_ms = inner.alpha * itl + (1.0 - inner.alpha) * inner.ema_itl_ms;
            }
        }
    }

    /// Produce a metrics snapshot for the SSE stream.
    pub fn snapshot(&self, kv_cache_utilization: f64, worker_heartbeats: Vec<u64>) -> MetricsEvent {
        let mut inner = self.inner.lock().unwrap();

        // Prune old entries from the throughput window.
        let cutoff = Instant::now() - std::time::Duration::from_secs_f64(inner.window_secs);
        while inner
            .token_window
            .front()
            .is_some_and(|(t, _)| *t < cutoff)
        {
            inner.token_window.pop_front();
        }

        let total_tokens_in_window: usize = inner.token_window.iter().map(|(_, n)| n).sum();
        let throughput = total_tokens_in_window as f64 / inner.window_secs;

        MetricsEvent {
            timestamp: chrono_iso8601_now(),
            throughput_tokens_per_sec: throughput,
            active_requests: inner.active_requests,
            avg_time_to_first_token_ms: inner.ema_ttft_ms,
            avg_inter_token_latency_ms: inner.ema_itl_ms,
            kv_cache_utilization,
            worker_heartbeats,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

fn chrono_iso8601_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    // Simple ISO 8601 without pulling in chrono crate.
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        1970 + secs / 31_556_952, // approximate year
        // For dashboard purposes, an approximate timestamp is fine.
        // Real ISO 8601 would need a full calendar library.
        (secs % 31_556_952) / 2_629_746 + 1,
        (secs % 2_629_746) / 86400 + 1,
        (secs % 86400) / 3600,
        (secs % 3600) / 60,
        secs % 60,
        millis,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(completion_tokens: usize, ttft_ms: f64, total_ms: f64) -> RequestRecord {
        RequestRecord {
            id: "test".to_string(),
            request_type: "chat",
            status: "completed",
            prompt_tokens: 10,
            completion_tokens,
            total_tokens: 10 + completion_tokens,
            time_to_first_token_ms: ttft_ms,
            total_duration_ms: total_ms,
            tokens_per_second: if total_ms > 0.0 {
                completion_tokens as f64 / (total_ms / 1000.0)
            } else {
                0.0
            },
            finish_reason: "stop",
            temperature: 0.7,
            created_at: String::new(),
        }
    }

    #[test]
    fn test_new_collector_has_zero_metrics() {
        let c = MetricsCollector::new();
        let snap = c.snapshot(0.0, vec![]);
        assert_eq!(snap.throughput_tokens_per_sec, 0.0);
        assert_eq!(snap.active_requests, 0);
        assert_eq!(snap.avg_time_to_first_token_ms, 0.0);
        assert_eq!(snap.avg_inter_token_latency_ms, 0.0);
    }

    #[test]
    fn test_active_request_tracking() {
        let c = MetricsCollector::new();
        c.request_started();
        c.request_started();
        assert_eq!(c.snapshot(0.0, vec![]).active_requests, 2);

        c.record_completion(&make_record(10, 100.0, 500.0));
        assert_eq!(c.snapshot(0.0, vec![]).active_requests, 1);

        c.record_completion(&make_record(5, 50.0, 200.0));
        assert_eq!(c.snapshot(0.0, vec![]).active_requests, 0);
    }

    #[test]
    fn test_active_request_saturating_sub() {
        let c = MetricsCollector::new();
        // Decrement without prior increment should not underflow.
        c.record_completion(&make_record(10, 100.0, 500.0));
        assert_eq!(c.snapshot(0.0, vec![]).active_requests, 0);
    }

    #[test]
    fn test_throughput_records_tokens() {
        let c = MetricsCollector::new();
        c.record_completion(&make_record(100, 50.0, 1000.0));
        let snap = c.snapshot(0.0, vec![]);
        // 100 tokens in a 10s window → 10 tok/s
        assert!(snap.throughput_tokens_per_sec > 0.0);
        assert!((snap.throughput_tokens_per_sec - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_ema_ttft_first_value_is_exact() {
        let c = MetricsCollector::new();
        c.record_completion(&make_record(10, 150.0, 500.0));
        let snap = c.snapshot(0.0, vec![]);
        assert!((snap.avg_time_to_first_token_ms - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_ema_ttft_smooths_over_multiple() {
        let c = MetricsCollector::new();
        c.record_completion(&make_record(10, 100.0, 500.0));
        c.record_completion(&make_record(10, 200.0, 500.0));
        let snap = c.snapshot(0.0, vec![]);
        // alpha=0.3: first=100, second = 0.3*200 + 0.7*100 = 130
        assert!((snap.avg_time_to_first_token_ms - 130.0).abs() < 0.01);
    }

    #[test]
    fn test_ema_itl_computed_from_generation_time() {
        let c = MetricsCollector::new();
        // 10 tokens, ttft=100ms, total=1000ms → generation=900ms, itl=900/9=100ms
        c.record_completion(&make_record(10, 100.0, 1000.0));
        let snap = c.snapshot(0.0, vec![]);
        assert!((snap.avg_inter_token_latency_ms - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_passes_through_kv_and_heartbeats() {
        let c = MetricsCollector::new();
        let snap = c.snapshot(0.75, vec![100, 200]);
        assert_eq!(snap.kv_cache_utilization, 0.75);
        assert_eq!(snap.worker_heartbeats, vec![100, 200]);
    }

    #[test]
    fn test_snapshot_has_timestamp() {
        let c = MetricsCollector::new();
        let snap = c.snapshot(0.0, vec![]);
        assert!(!snap.timestamp.is_empty());
        assert!(snap.timestamp.contains('T'));
        assert!(snap.timestamp.ends_with('Z'));
    }
}
