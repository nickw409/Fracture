use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tower_http::services::ServeDir;

use super::dto::*;
use super::metrics::MetricsCollector;
use super::request_log::RequestLog;
use crate::scheduler_loop::SchedulerHandle;

/// Mode-specific cluster information.
pub enum ClusterProvider {
    /// Single GPU node (standalone or batched standalone).
    Standalone {
        gpu_name: String,
        vram_total_mb: u64,
        vram_used_mb: u64,
        model: ModelInfo,
        total_layers: usize,
    },
    /// Live cluster state updated externally (e.g., by the coordinator's heartbeat loop).
    /// The watch receiver always has the latest snapshot.
    Live(tokio::sync::watch::Receiver<ClusterResponse>),
}

/// Shared state for all dashboard endpoints.
pub struct DashboardState {
    pub metrics: Arc<MetricsCollector>,
    pub request_log: Arc<RequestLog>,
    pub cluster: ClusterProvider,
    pub scheduler: Option<SchedulerHandle>,
}

/// Create the dashboard sub-router (API endpoints + optional static file serving).
pub fn dashboard_routes(state: Arc<DashboardState>) -> Router {
    let mut router = Router::new()
        .route("/v1/cluster", get(cluster_handler))
        .route("/v1/scheduler", get(scheduler_handler))
        .route("/v1/metrics/stream", get(metrics_stream_handler))
        .route("/v1/requests", get(requests_handler))
        .with_state(state);

    // Serve the dashboard frontend if the dist directory exists.
    let dist_path = Path::new("fracture-dashboard/dist");
    if dist_path.exists() {
        tracing::info!("serving dashboard UI at /dashboard from {}", dist_path.display());
        router = router.nest_service("/dashboard", ServeDir::new(dist_path));
    }

    router
}

// ── GET /v1/cluster ──────────────────────────────────────

async fn cluster_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let response = match &state.cluster {
        ClusterProvider::Standalone {
            gpu_name,
            vram_total_mb,
            vram_used_mb,
            model,
            total_layers,
        } => ClusterResponse {
            mode: "standalone",
            num_workers: 1,
            workers: vec![WorkerInfo {
                id: 0,
                role: "standalone",
                address: "local".to_string(),
                gpu: gpu_name.clone(),
                vram_total_mb: *vram_total_mb,
                vram_used_mb: *vram_used_mb,
                layers: [0, total_layers.saturating_sub(1)],
                status: "active",
                last_heartbeat_ms: 0,
                calibration_ms_per_layer: 0.0,
            }],
            scheduling_mode: "auto",
            model: model.clone(),
        },
        ClusterProvider::Live(rx) => rx.borrow().clone(),
    };

    (StatusCode::OK, Json(response))
}

// ── GET /v1/scheduler ────────────────────────────────────

async fn scheduler_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let response = if let Some(ref handle) = state.scheduler {
        match handle.snapshot().await {
            Ok(snap) => snap.to_response(),
            Err(_) => empty_scheduler_response(),
        }
    } else {
        empty_scheduler_response()
    };

    (StatusCode::OK, Json(response))
}

fn empty_scheduler_response() -> SchedulerResponse {
    SchedulerResponse {
        active_sequences: 0,
        max_sequences: 0,
        decode_queue: 0,
        prefill_queue: 0,
        prefill_chunk_size: 0,
        kv_cache: KvCacheInfo {
            block_size: 16,
            total_blocks: 0,
            allocated_blocks: 0,
            free_blocks: 0,
            utilization: 0.0,
        },
        sequences: vec![],
    }
}

// ── GET /v1/metrics/stream ───────────────────────────────

async fn metrics_stream_handler(
    State(state): State<Arc<DashboardState>>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            // Pull cache utilization from scheduler snapshot or cluster data.
            let (kv_util, heartbeats) = if let Some(ref handle) = state.scheduler {
                let util = handle.snapshot().await
                    .map(|s| {
                        if s.total_blocks > 0 {
                            (s.total_blocks - s.free_blocks) as f64 / s.total_blocks as f64
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);
                (util, vec![0])
            } else if let ClusterProvider::Live(ref rx) = state.cluster {
                let cluster = rx.borrow();
                let heartbeats: Vec<u64> = cluster.workers.iter().map(|w| w.last_heartbeat_ms).collect();
                // TODO: pipe free_blocks from heartbeat acks for real cache utilization.
                // For now report 0 — VRAM usage is misleading (includes model weights).
                (0.0, heartbeats)
            } else {
                (0.0, vec![0])
            };
            let snapshot = state.metrics.snapshot(kv_util, heartbeats);
            let data = serde_json::to_string(&snapshot).unwrap_or_default();
            yield Ok::<_, Infallible>(Event::default().data(data));
        }
    };

    Sse::new(stream)
}

// ── GET /v1/requests ─────────────────────────────────────

#[derive(Deserialize)]
struct PaginationParams {
    page: Option<usize>,
    per_page: Option<usize>,
}

async fn requests_handler(
    State(state): State<Arc<DashboardState>>,
    Query(params): Query<PaginationParams>,
) -> impl IntoResponse {
    let response = state
        .request_log
        .page(params.page.unwrap_or(1), params.per_page.unwrap_or(50));
    (StatusCode::OK, Json(response))
}
