//! Admin API endpoints for cluster management.
//!
//! - POST /admin/rebalance — trigger immediate forced rebalance
//! - GET  /admin/cluster   — return cluster state
//! - POST /admin/drain     — trigger graceful rebalance (wait for drain)

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use fracture_coordinator::registry::PeerRegistry;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared state for admin endpoints.
pub struct AdminState {
    pub registry: Arc<Mutex<PeerRegistry>>,
    /// Signal to the distributed loop to trigger a forced rebalance.
    pub rebalance_tx: tokio::sync::mpsc::UnboundedSender<RebalanceRequest>,
}

/// Type of rebalance requested via admin API.
#[derive(Debug, Clone)]
pub enum RebalanceRequest {
    /// Immediate forced rebalance (aborts active sequences).
    Forced,
    /// Graceful rebalance (waits for active sequences to drain).
    Graceful,
}

/// GET /admin/cluster — return cluster state.
pub async fn cluster_handler(
    State(state): State<Arc<AdminState>>,
) -> impl IntoResponse {
    let reg = state.registry.lock().await;
    let workers: Vec<ClusterWorkerInfo> = reg
        .iter()
        .map(|(id, entry)| ClusterWorkerInfo {
            node_id: id.clone(),
            status: format!("{:?}", entry.status),
            gpu_model: entry.capabilities.gpu_model.clone(),
            layers: entry
                .assignment
                .as_ref()
                .map(|a| format!("{:?}", a.layer_range)),
            free_blocks: entry.free_blocks,
            gpu_memory_used_mb: entry.gpu_memory_used / (1024 * 1024),
        })
        .collect();
    let pending_count = reg.pending_workers().len();
    drop(reg);

    Json(ClusterStateResponse {
        num_workers: workers.len(),
        workers,
        pending_joins: pending_count,
    })
}

/// POST /admin/rebalance — trigger immediate forced rebalance.
pub async fn rebalance_handler(
    State(state): State<Arc<AdminState>>,
) -> impl IntoResponse {
    match state.rebalance_tx.send(RebalanceRequest::Forced) {
        Ok(()) => (StatusCode::OK, Json(AdminResponse {
            status: "ok".into(),
            message: "forced rebalance triggered".into(),
        })),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(AdminResponse {
            status: "error".into(),
            message: "distributed loop not running".into(),
        })),
    }
}

/// POST /admin/drain — trigger graceful rebalance (wait for drain).
pub async fn drain_handler(
    State(state): State<Arc<AdminState>>,
) -> impl IntoResponse {
    match state.rebalance_tx.send(RebalanceRequest::Graceful) {
        Ok(()) => (StatusCode::OK, Json(AdminResponse {
            status: "ok".into(),
            message: "graceful rebalance triggered (will execute when active sequences drain)".into(),
        })),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(AdminResponse {
            status: "error".into(),
            message: "distributed loop not running".into(),
        })),
    }
}

#[derive(Serialize)]
pub struct ClusterStateResponse {
    pub num_workers: usize,
    pub workers: Vec<ClusterWorkerInfo>,
    pub pending_joins: usize,
}

#[derive(Serialize)]
pub struct ClusterWorkerInfo {
    pub node_id: String,
    pub status: String,
    pub gpu_model: String,
    pub layers: Option<String>,
    pub free_blocks: u32,
    pub gpu_memory_used_mb: u64,
}

#[derive(Serialize)]
pub struct AdminResponse {
    pub status: String,
    pub message: String,
}

/// Build the admin router.
pub fn admin_routes(state: Arc<AdminState>) -> axum::Router {
    axum::Router::new()
        .route("/admin/cluster", axum::routing::get(cluster_handler))
        .route("/admin/rebalance", axum::routing::post(rebalance_handler))
        .route("/admin/drain", axum::routing::post(drain_handler))
        .with_state(state)
}
