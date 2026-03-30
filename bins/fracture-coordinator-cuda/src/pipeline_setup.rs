use anyhow::Result;
use fracture_coordinator::{
    pipeline::DistributedPipeline,
    registry::PeerRegistry,
    scheduler::{self, SchedulerInput, SchedulingMode, WorkerCapabilities},
};
use fracture_protocol::{
    connection::FramedConnection,
    frame::MessageType,
    messages::*,
};
use fracture_server::dashboard::dto::{
    ClusterResponse, ModelInfo, WorkerInfo,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

/// Accept workers, run scheduler, set up pipeline, and broadcast via watch channel.
#[allow(clippy::too_many_arguments)]
pub async fn accept_and_setup_pipeline(
    listener: &TcpListener,
    registry: &Mutex<PeerRegistry>,
    model_config: &fracture_core::ModelConfig,
    scheduling_mode: SchedulingMode,
    expected_workers: usize,
    max_seq_len: usize,
    acceptance_timeout_secs: u64,
    pipeline_tx: &tokio::sync::watch::Sender<Arc<DistributedPipeline>>,
) -> Result<()> {
    let timeout_duration = if acceptance_timeout_secs > 0 {
        Some(Duration::from_secs(acceptance_timeout_secs))
    } else {
        None
    };
    tracing::info!(
        "waiting for {} workers to register (timeout: {})...",
        expected_workers,
        timeout_duration.map_or("none".to_string(), |d| format!("{}s", d.as_secs()))
    );

    let accept_start = std::time::Instant::now();

    loop {
        {
            let reg = registry.lock().await;
            if reg.active_count() >= expected_workers {
                break;
            }
        }

        if let Some(timeout) = timeout_duration
            && accept_start.elapsed() >= timeout {
                let reg = registry.lock().await;
                anyhow::bail!(
                    "timed out waiting for workers: got {}/{} after {}s",
                    reg.active_count(),
                    expected_workers,
                    timeout.as_secs()
                );
            }

        let accept_future = listener.accept();
        let (stream, addr) = if let Some(timeout) = timeout_duration {
            let remaining = timeout.saturating_sub(accept_start.elapsed());
            match tokio::time::timeout(remaining, accept_future).await {
                Ok(result) => result?,
                Err(_) => {
                    let reg = registry.lock().await;
                    anyhow::bail!(
                        "timed out waiting for workers: got {}/{} after {}s",
                        reg.active_count(),
                        expected_workers,
                        acceptance_timeout_secs
                    );
                }
            }
        } else {
            accept_future.await?
        };
        tracing::info!("worker connected from {addr}");
        let mut conn = FramedConnection::new(stream);

        let (header, payload) = conn.recv().await?;
        if header.msg_type != MessageType::Register {
            tracing::warn!(
                "expected Register from {addr}, got {:?} — dropping",
                header.msg_type
            );
            continue;
        }

        let reg_msg: RegisterPayload = FramedConnection::deserialize_payload(&payload)?;
        tracing::info!(
            "worker '{}' registered: {} ({:.1} GB available, decode={:.2} ms/layer)",
            reg_msg.node_id,
            reg_msg.gpu_model,
            reg_msg.gpu_memory_available as f64 / 1e9,
            reg_msg.decode_ms_per_layer
        );

        let caps = WorkerCapabilities {
            node_id: reg_msg.node_id.clone(),
            gpu_model: reg_msg.gpu_model,
            gpu_memory_available: reg_msg.gpu_memory_available as usize,
            compute_capability: reg_msg.compute_capability,
            decode_ms_per_layer: reg_msg.decode_ms_per_layer,
            prefill_ms_per_layer_128: reg_msg.prefill_ms_per_layer_128,
        };

        let mut reg = registry.lock().await;
        reg.register(caps, conn)?;
    }

    tracing::info!("all {} workers registered — running scheduler", expected_workers);

    // Run scheduler.
    let mut reg = registry.lock().await;
    let scheduler_input = SchedulerInput {
        model_config: model_config.clone(),
        workers: reg.all_capabilities(),
        coordinator_compute: None,
        mode: scheduling_mode,
        max_seq_len,
        hop_latency_ms: 2.0,
    };

    let schedule_result = scheduler::schedule(&scheduler_input)?;

    tracing::info!("scheduler result:");
    for a in &schedule_result.assignments {
        tracing::info!(
            "  {} → layers {:?} ({:?}), {:.1} ms/decode, {:.1} GB weights, {:.1} GB cache",
            a.node_id, a.layer_range, a.role, a.expected_decode_ms, a.weight_memory_gb, a.cache_memory_gb
        );
    }

    // Send RegisterAck to each worker.
    for assignment in &schedule_result.assignments {
        let ack = RegisterAckPayload {
            layer_start: assignment.layer_range.start as u32,
            layer_end: assignment.layer_range.end as u32,
            total_layers: model_config.num_layers as u32,
            max_seq_len: max_seq_len as u32,
            model_config: model_config.clone(),
        };
        reg.assign(&assignment.node_id, assignment.clone())?;
        let entry = reg.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared", assignment.node_id)
        })?;
        entry.writer.send(MessageType::RegisterAck, 0, &ack).await?;
        tracing::info!("sent RegisterAck to '{}'", assignment.node_id);
    }

    // Wait for WorkerReady from each.
    tracing::info!("waiting for workers to finish weight loading...");
    for assignment in &schedule_result.assignments {
        let entry = reg.get_mut(&assignment.node_id).ok_or_else(|| {
            anyhow::anyhow!("worker '{}' disappeared", assignment.node_id)
        })?;
        let (header, _) = entry.reader.lock().await.recv().await?;
        if header.msg_type != MessageType::WorkerReady {
            anyhow::bail!(
                "expected WorkerReady from '{}', got {:?}",
                assignment.node_id, header.msg_type
            );
        }
        tracing::info!("worker '{}' ready", assignment.node_id);
    }
    drop(reg);

    // Build pipeline and broadcast.
    let pipeline = Arc::new(DistributedPipeline::new(
        &schedule_result.assignments,
        model_config.hidden_size,
    )?);
    tracing::info!("distributed pipeline ready with {} stages", pipeline.pipeline_order().len());
    let _ = pipeline_tx.send(pipeline);

    Ok(())
}

pub async fn build_cluster_snapshot(
    registry: &Mutex<PeerRegistry>,
    model_config: &fracture_core::ModelConfig,
    max_seq_len: usize,
) -> ClusterResponse {
    let reg = registry.lock().await;
    let order = reg.pipeline_order();
    let workers: Vec<WorkerInfo> = order
        .iter()
        .enumerate()
        .filter_map(|(i, node_id)| {
            let entry = reg.get(node_id)?;
            let (layer_start, layer_end) = entry
                .assignment
                .as_ref()
                .map(|a| (a.layer_range.start, a.layer_range.end.saturating_sub(1)))
                .unwrap_or((0, 0));
            let role = if i == 0 {
                "head"
            } else if i == order.len() - 1 {
                "tail"
            } else {
                "middle"
            };
            let status = match entry.status {
                fracture_coordinator::registry::WorkerStatus::Connected => "calibrating",
                fracture_coordinator::registry::WorkerStatus::Ready => "active",
                fracture_coordinator::registry::WorkerStatus::Draining => "draining",
                fracture_coordinator::registry::WorkerStatus::Pending => "pending",
                fracture_coordinator::registry::WorkerStatus::Dead => "dead",
            };
            let vram_total_mb =
                (entry.capabilities.gpu_memory_available / (1024 * 1024)) as u64;
            let vram_used_mb = entry.gpu_memory_used / (1024 * 1024);
            let heartbeat_age_ms = entry.last_heartbeat.elapsed().as_millis() as u64;
            Some(WorkerInfo {
                id: i,
                role,
                address: entry.capabilities.node_id.clone(),
                gpu: entry.capabilities.gpu_model.clone(),
                vram_total_mb,
                vram_used_mb,
                layers: [layer_start, layer_end],
                status,
                last_heartbeat_ms: heartbeat_age_ms,
                calibration_ms_per_layer: entry.capabilities.decode_ms_per_layer as f64,
            })
        })
        .collect();
    ClusterResponse {
        mode: "distributed",
        num_workers: workers.len(),
        workers,
        scheduling_mode: "auto",
        model: ModelInfo {
            name: "llama-3-8b".to_string(),
            parameters: "8B".to_string(),
            layers: model_config.num_layers,
            context_length: max_seq_len,
            dtype: "FP16".to_string(),
        },
    }
}

pub fn load_tokenizer(model_path: &str, tokenizer_path: Option<&str>) -> Result<Tokenizer> {
    if let Some(path) = tokenizer_path {
        return Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"));
    }
    let model_dir = std::path::Path::new(model_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let tokenizer_file = model_dir.join("tokenizer.json");
    if tokenizer_file.exists() {
        Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
    } else {
        anyhow::bail!(
            "no tokenizer found. Provide --tokenizer <path> or place tokenizer.json next to the model file"
        );
    }
}

/// Background task that accepts new worker connections for reconnection.
///
/// Workers that previously died can reconnect by opening a new TCP connection
/// and sending Register. This task processes the registration and triggers
/// pipeline reconfiguration.
pub async fn reconnection_listener(
    listener: TcpListener,
    registry: Arc<Mutex<PeerRegistry>>,
    model_config: fracture_core::ModelConfig,
    scheduling_mode: SchedulingMode,
    max_seq_len: usize,
    pipeline_tx: tokio::sync::watch::Sender<Arc<DistributedPipeline>>,
    // Coordinator's own address for seed discovery responses.
    self_addr: String,
) {
    loop {
        let (stream, addr) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("reconnection listener accept error: {e}");
                continue;
            }
        };

        tracing::info!("worker reconnecting from {addr}");
        let mut conn = FramedConnection::new(stream);

        let (header, payload) = match conn.recv().await {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!("failed to read Register from {addr}: {e}");
                continue;
            }
        };

        // Handle seed discovery queries: respond with our address and continue listening.
        if header.msg_type == MessageType::WhoIsCoordinator {
            tracing::info!("seed discovery query from {addr}");
            let resp = WhoIsCoordinatorResponsePayload {
                coordinator_addr: Some(self_addr.clone()),
                term: 0, // TODO: track coordinator term
                manifest: None, // TODO: include manifest when available
            };
            let _ = conn.send(MessageType::WhoIsCoordinatorResponse, 0, &resp).await;
            continue;
        }

        // Accept both Register (fresh worker process) and ReRegister (surviving worker).
        let (node_id, caps, is_reregister, current_assignment) = match header.msg_type {
            MessageType::Register => {
                let reg_msg: RegisterPayload = match FramedConnection::deserialize_payload(&payload) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("failed to deserialize Register from {addr}: {e}");
                        continue;
                    }
                };
                tracing::info!(
                    "reconnecting worker '{}' (fresh process): {} ({:.1} GB)",
                    reg_msg.node_id,
                    reg_msg.gpu_model,
                    reg_msg.gpu_memory_available as f64 / 1e9,
                );
                let caps = WorkerCapabilities {
                    node_id: reg_msg.node_id.clone(),
                    gpu_model: reg_msg.gpu_model,
                    gpu_memory_available: reg_msg.gpu_memory_available as usize,
                    compute_capability: reg_msg.compute_capability,
                    decode_ms_per_layer: reg_msg.decode_ms_per_layer,
                    prefill_ms_per_layer_128: reg_msg.prefill_ms_per_layer_128,
                };
                (reg_msg.node_id, caps, false, None)
            }
            MessageType::ReRegister => {
                let rereg: ReRegisterPayload = match FramedConnection::deserialize_payload(&payload) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("failed to deserialize ReRegister from {addr}: {e}");
                        continue;
                    }
                };
                let current = rereg.current_layer_start.zip(rereg.current_layer_end)
                    .map(|(s, e)| s as usize..e as usize);
                tracing::info!(
                    "reconnecting worker '{}' (surviving process): layers {:?}, {} active caches",
                    rereg.node_id,
                    current,
                    rereg.active_cache_seq_ids.len(),
                );
                let caps = WorkerCapabilities {
                    node_id: rereg.node_id.clone(),
                    gpu_model: rereg.gpu_model,
                    gpu_memory_available: rereg.gpu_memory_available as usize,
                    compute_capability: rereg.compute_capability,
                    decode_ms_per_layer: rereg.decode_ms_per_layer,
                    prefill_ms_per_layer_128: rereg.prefill_ms_per_layer_128,
                };
                (rereg.node_id, caps, true, current)
            }
            other => {
                tracing::warn!(
                    "expected Register or ReRegister from {addr}, got {:?}",
                    other
                );
                continue;
            }
        };

        // Check if this is a truly new worker (never seen before) or a reconnection.
        let is_new_worker = {
            let reg = registry.lock().await;
            reg.get(&node_id).is_none()
        };

        {
            let mut reg = registry.lock().await;
            if !is_new_worker {
                // Mark the old entry as dead so re-registration succeeds.
                reg.mark_dead(&node_id);
            }
            if let Err(e) = reg.register(caps, conn) {
                tracing::error!("failed to register '{}': {e}", node_id);
                continue;
            }
            // New workers are held in Pending state until the distributed loop
            // incorporates them via graceful rebalance.
            if is_new_worker {
                reg.mark_pending(&node_id);
                tracing::info!(
                    "new worker '{}' registered as Pending — will join pipeline after current sequences drain",
                    node_id
                );
                continue;
            }
        }

        // Reconnecting worker: re-run scheduler and reconfigure immediately.
        let result = {
            let reg = registry.lock().await;
            let all_caps = reg.all_capabilities();
            if all_caps.is_empty() {
                tracing::error!("no workers available after re-registration");
                continue;
            }
            let input = scheduler::SchedulerInput {
                model_config: model_config.clone(),
                workers: all_caps,
                coordinator_compute: None,
                mode: scheduling_mode.clone(),
                max_seq_len,
                hop_latency_ms: 2.0,
            };
            match scheduler::schedule(&input) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("scheduler failed after reconnect: {e}");
                    continue;
                }
            }
        };

        tracing::info!("new schedule after reconnect:");
        for a in &result.assignments {
            tracing::info!("  {} → layers {:?} ({:?})", a.node_id, a.layer_range, a.role);
        }

        // Send response to each worker. Logic depends on whether the reconnecting
        // worker used Register (fresh process) or ReRegister (surviving process).
        let mut all_ok = true;
        {
            let mut reg = registry.lock().await;
            for assignment in &result.assignments {
                let ack = RegisterAckPayload {
                    layer_start: assignment.layer_range.start as u32,
                    layer_end: assignment.layer_range.end as u32,
                    total_layers: model_config.num_layers as u32,
                    max_seq_len: max_seq_len as u32,
                    model_config: model_config.clone(),
                };
                reg.assign(&assignment.node_id, assignment.clone()).ok();
                if let Some(entry) = reg.get_mut(&assignment.node_id) {
                    let msg_type = if assignment.node_id == node_id {
                        if is_reregister {
                            // Surviving worker: check if assignment is unchanged.
                            let assignment_unchanged = current_assignment.as_ref()
                                .is_some_and(|cur| *cur == assignment.layer_range);
                            if assignment_unchanged {
                                // Same layers — worker can skip weight reload.
                                MessageType::RegisterAck
                            } else {
                                // Different layers — worker must reconfigure.
                                MessageType::Reconfigure
                            }
                        } else {
                            // Fresh process always gets RegisterAck.
                            MessageType::RegisterAck
                        }
                    } else {
                        // Other workers always get Reconfigure.
                        MessageType::Reconfigure
                    };
                    if let Err(e) = entry.writer.send(msg_type, 0, &ack).await {
                        tracing::error!("failed to send to '{}': {e}", assignment.node_id);
                        all_ok = false;
                    }
                }
            }
        }

        if !all_ok {
            tracing::error!("reconfiguration after reconnect partially failed");
            continue;
        }

        // Wait for WorkerReady from all workers.
        {
            let reg = registry.lock().await;
            for assignment in &result.assignments {
                if let Some(entry) = reg.get(&assignment.node_id) {
                    match entry.reader.lock().await.recv().await {
                        Ok((hdr, _)) if hdr.msg_type == MessageType::WorkerReady => {
                            tracing::info!("worker '{}' ready after reconnect", assignment.node_id);
                        }
                        Ok((hdr, _)) => {
                            tracing::error!(
                                "expected WorkerReady from '{}', got {:?}",
                                assignment.node_id, hdr.msg_type
                            );
                        }
                        Err(e) => {
                            tracing::error!("recv from '{}' failed: {e}", assignment.node_id);
                        }
                    }
                }
            }
        }

        // Build new pipeline and broadcast.
        match DistributedPipeline::new(&result.assignments, model_config.hidden_size) {
            Ok(new_pipeline) => {
                let _ = pipeline_tx.send(Arc::new(new_pipeline));
                tracing::info!("pipeline reconfigured after worker reconnection");
            }
            Err(e) => {
                tracing::error!("failed to build pipeline after reconnect: {e}");
            }
        }
    }
}
