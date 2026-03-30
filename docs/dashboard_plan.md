# Fracture Dashboard Implementation Plan

## Context

Adding a React + TypeScript observability dashboard to Fracture as a portfolio piece. The dashboard connects to new and existing axum REST/SSE endpoints, showing cluster topology, live metrics, an inference playground, and scheduler state. Spec: `docs/dashboard.md`.

---

## Architecture Decisions

1. **Dashboard DTO types** go in a new `dashboard` module in `fracture-server` — separate from internal types, all derive `Serialize`.
2. **MetricsCollector** and **RequestLog** are `Arc`-shared structs added to `BatchedAppState` / `AppState` / `CoordState`. They're populated in the HTTP handler layer (not inside the scheduler loop), keeping the hot path clean.
3. **Scheduler snapshot** uses a oneshot channel injected into the scheduler loop's command channel — avoids wrapping `BatchScheduler` in a shared Mutex.
4. **Frontend mock mode** (`VITE_MOCK=true`) provides realistic fake data so the dashboard is runnable without a GPU or model.
5. **CORS** via `tower-http::CorsLayer::permissive()` on all routers (dev only; production serves from same origin).

---

## Implementation Order

### Phase 0: Backend Infrastructure

**Step 0.1 — DTO types**
- Create `crates/fracture-server/src/dashboard/mod.rs`, `dto.rs`
- Define all `#[derive(Serialize)]` response structs: `ClusterResponse`, `WorkerInfo`, `SchedulerResponse`, `SequenceInfo`, `MetricsEvent`, `RequestsResponse`, `RequestRecord`
- Modify `crates/fracture-server/src/lib.rs` to add `pub mod dashboard`

**Step 0.2 — MetricsCollector**
- Create `crates/fracture-server/src/dashboard/metrics.rs`
- Sliding 10s window for throughput, EMA for TTFT/ITL
- `record_completion(record)` and `snapshot() -> MetricsEvent` methods
- `Arc<MetricsCollector>` for sharing

**Step 0.3 — RequestLog**
- Create `crates/fracture-server/src/dashboard/request_log.rs`
- Bounded `VecDeque<RequestRecord>` (cap 1000) behind `Mutex`
- `push(record)`, `page(page, per_page) -> (Vec<RequestRecord>, total)`

**Step 0.4 — Wire into request completion**
- Modify `batched_routes.rs`: add `metrics: Arc<MetricsCollector>` and `request_log: Arc<RequestLog>` to `BatchedAppState`. After streaming finishes, build `RequestRecord` from timing data and push to both.
- Modify `routes.rs`: same for `AppState<B>` (add to struct, record on completion)
- Modify coordinator `main.rs`: same for `CoordState`

**Step 0.5 — Scheduler snapshot mechanism**
- Modify `scheduler_loop.rs`: change internal channel to `enum SchedulerCommand { Submit(PendingRequest), Snapshot(oneshot::Sender<SchedulerSnapshotDto>) }`. Add `SchedulerHandle::snapshot()` async method. In loop, handle snapshot by reading `BatchScheduler` + `PagedKvCacheManager` state.
- Modify `distributed_loop.rs`: same pattern for distributed scheduler

### Phase 1: Backend Endpoints

**Step 1.1 — CORS middleware**
- Add `.layer(CorsLayer::permissive())` to `create_router()`, `create_batched_router()`, and coordinator router

**Step 1.2 — `GET /v1/cluster`**
- Create `crates/fracture-server/src/dashboard/routes.rs`
- Standalone: single worker from backend GPU info + model config
- Distributed: serialize from `PeerRegistry` + assignments
- Add `DashboardState` enum for mode-specific cluster info, passed alongside existing app state

**Step 1.3 — `GET /v1/scheduler`**
- Uses `SchedulerHandle::snapshot()` (batched modes)
- Standalone non-batched: return empty/zero response

**Step 1.4 — `GET /v1/metrics/stream`**
- SSE via `axum::response::sse::Sse` + `async_stream`
- `tokio::time::interval(1s)` tick, calls `MetricsCollector::snapshot()`
- Always emits (zero values when idle)

**Step 1.5 — `GET /v1/requests`**
- Reads from `RequestLog::page()` with `Query<PaginationParams>` extractor

**Step 1.6 — Merge dashboard routes into routers**
- Create `dashboard_routes(state) -> Router` function
- Merge into `create_router()`, `create_batched_router()`, and coordinator router
- Add `ServeDir` for `/dashboard` → `fracture-dashboard/dist/`

### Phase 2: Frontend Foundation

**Step 2.1 — Scaffold project**
- `fracture-dashboard/`: Vite + React 18 + TypeScript (strict) + Tailwind CSS
- Dependencies: recharts, react-router-dom v6, lucide-react
- Vite proxy: `/v1` and `/health` → `http://localhost:8080`

**Step 2.2 — API layer**
- `src/api/types.ts` — all TypeScript interfaces
- `src/api/client.ts` — typed fetch wrapper
- `src/api/sse.ts` — POST-based SSE streaming for chat

**Step 2.3 — Utilities**
- `src/lib/formatters.ts` — formatTokens, formatDuration, formatBytes, formatPercent
- `src/lib/constants.ts` — polling intervals, buffer sizes

**Step 2.4 — Layout shell + routing**
- `Shell.tsx`, `Sidebar.tsx`, `StatusBar.tsx`
- `App.tsx` with React Router: `/` (Cluster), `/playground`, `/scheduler`
- Three placeholder pages

**Step 2.5 — Mock data mode**
- `src/api/mock.ts` — generators for all API responses with realistic values
- Hooks check `VITE_MOCK=true` and return mock data

### Phase 3: Cluster Page

- `useCluster` hook (5s polling)
- `useMetricsStream` hook (SSE, 60-event rolling buffer)
- `Sparkline`, `ProgressBar`, `StatusDot`, `Skeleton` shared components
- `WorkerCard` — GPU name, layer range, VRAM bar, heartbeat dot, calibration speed
- `PipelineView` — horizontal worker chain with CSS/SVG arrow connectors
- `ModelInfo` — model metadata card
- `MetricsSummary` — 4 stat cards with sparklines (throughput, active requests, TTFT, cache util)
- Assemble `ClusterPage`

### Phase 4: Inference Playground

- `useInference` hook — manages messages, streaming, abort, timing stats
- `MessageBubble` — user (right, blue tint) / assistant (left, gray)
- `ChatView` — message list + textarea (Enter to send, Shift+Enter newline), streaming cursor, abort button
- `ParamControls` — sliders/inputs for temperature, top_p, top_k, max_tokens, seed
- `GenerationStats` — TTFT, tok/s, total tokens, finish reason
- Assemble `PlaygroundPage` (two-column layout)

### Phase 5: Scheduler Page

- `useScheduler` hook (2s polling)
- `useRequests` hook (5s polling)
- `QueueOverview` — active sequences, decode queue, prefill queue stat cards
- `CachePool` — horizontal bar (emerald/amber/red by utilization), block stats
- `SequenceTable` — sortable table with colored state badges
- Assemble `SchedulerPage`

### Phase 6: Integration & Polish

- Connect frontend to real backend (remove mock default)
- `StatusBar` connection indicator (green/red dot via `/health` polling)
- `useMetricsStream` reconnection with 3s backoff
- Error cards for failed API calls
- Build and test static file serving at `/dashboard`

---

## Key Files to Modify (Backend)

| File | Change |
|---|---|
| `crates/fracture-server/src/lib.rs` | Add `pub mod dashboard` |
| `crates/fracture-server/src/batched_routes.rs` | Add metrics/request_log to `BatchedAppState`, record timing |
| `crates/fracture-server/src/routes.rs` | Add metrics/request_log to `AppState`, record timing, CORS |
| `crates/fracture-server/src/scheduler_loop.rs` | `SchedulerCommand` enum, snapshot mechanism |
| `bins/fracture-server-cuda/src/main.rs` | Construct `DashboardState`, pass metrics/request_log |
| `bins/fracture-coordinator-cuda/src/main.rs` | Construct `DashboardState`, pass metrics/request_log |
| `bins/fracture-coordinator-cuda/src/distributed_loop.rs` | Snapshot mechanism |

## Key Files to Create (Backend)

| File | Purpose |
|---|---|
| `crates/fracture-server/src/dashboard/mod.rs` | Module root |
| `crates/fracture-server/src/dashboard/dto.rs` | All serializable response types |
| `crates/fracture-server/src/dashboard/metrics.rs` | MetricsCollector |
| `crates/fracture-server/src/dashboard/request_log.rs` | RequestLog |
| `crates/fracture-server/src/dashboard/routes.rs` | Dashboard endpoint handlers |

## Verification

1. **Backend only:** `cargo check` compiles. `cargo nextest run` passes (existing tests unbroken).
2. **Frontend only:** `cd fracture-dashboard && npm run dev` — dashboard renders with mock data, all three pages navigable.
3. **Integration:** Start Fracture server with a model, open dashboard — cluster shows real GPU, playground streams real tokens, scheduler reflects batch state.
4. **Production build:** `npm run build` produces `dist/`, served at `/dashboard` by the Rust server.

## Risks

- **Scheduler snapshot (Step 0.5)** is the highest-risk backend change — modifies the scheduler loop's channel type. Must not regress hot-path performance.
- **Standalone mode** has limited data (no block pool, single worker). Endpoints must handle gracefully.
- **Frontend without backend** must be fully functional via mock mode for development/demos.
