# DASHBOARD.md — Fracture Dashboard Implementation Guide

> This document is the single source of truth for building the Fracture dashboard frontend and its supporting backend endpoints. Reference this before implementing any component.

---

## Project Context

Fracture is a distributed LLM inference engine in Rust (~37k lines) with CUDA backends. It already has an OpenAI-compatible HTTP API served by axum. We are adding a React + TypeScript observability dashboard that connects to both existing and new backend endpoints.

The dashboard lives in `fracture-dashboard/` at the workspace root as a standalone Vite project. The Rust backend serves it as static files at `/dashboard` in production, and in dev mode the Vite dev server proxies API calls to the Rust backend.

**Purpose:** This is a portfolio piece for a job application at a YC startup. It needs to demonstrate TypeScript + React proficiency, real-time data handling, data visualization, complex state management, and full-stack integration. It should look and feel like a real product, not a tutorial project.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | React 18 | Industry standard, what the target company uses |
| Language | TypeScript (strict mode) | Required by target role |
| Build | Vite | Fast dev server, good TS support |
| Styling | Tailwind CSS | Utility-first, fast iteration |
| Charts | Recharts | React-native charting, composable |
| Routing | React Router v6 | Standard, supports nested layouts |
| SSE | Native EventSource + fetch for POST streams | No library needed |
| State | React context + useReducer for global, useState/useRef for local | No Redux, no Zustand — keep it simple |
| Icons | Lucide React | Clean, consistent icon set |

**No other dependencies unless strictly necessary.** Keep the dependency tree small.

---

## Design Language

Dark theme by default — this is an infrastructure dashboard, not a consumer app. Use a dark gray base with accent colors for status and data.

### Color Palette (Tailwind classes)

- **Background:** `bg-gray-950` (page), `bg-gray-900` (cards/panels), `bg-gray-800` (elevated elements)
- **Text:** `text-gray-100` (primary), `text-gray-400` (secondary/muted), `text-gray-500` (disabled)
- **Borders:** `border-gray-700` (subtle), `border-gray-600` (emphasized)
- **Accent green** (healthy/active): `text-emerald-400`, `bg-emerald-400/10`, `border-emerald-400/30`
- **Accent blue** (info/primary action): `text-blue-400`, `bg-blue-400/10`
- **Accent amber** (warning/pressure): `text-amber-400`, `bg-amber-400/10`
- **Accent red** (error/dead): `text-red-400`, `bg-red-400/10`
- **Accent purple** (GPU/compute): `text-violet-400`, `bg-violet-400/10`

### Typography

- Use `font-mono` for all numeric data, metrics, IDs, and code.
- Use default sans font for labels, descriptions, and prose.
- Don't over-size headings. Page titles: `text-xl font-semibold`. Section headers: `text-sm font-medium text-gray-400 uppercase tracking-wider`.

### Component Patterns

- **Cards**: `bg-gray-900 border border-gray-700 rounded-lg p-4`
- **Stat display**: Large mono number + small muted label below it
- **Status indicator**: Small colored dot (8px circle) + status text
- **Tables**: No zebra striping. Subtle `border-b border-gray-800` between rows. `text-sm`. Header row `text-gray-400 text-xs uppercase`.

### General Rules

- No rounded-full buttons. Use `rounded-md` or `rounded-lg`.
- No gradients. Flat colors only.
- Generous spacing. Don't cram elements. Use `gap-4` or `gap-6` between cards.
- All interactive elements need `transition-colors duration-150` for hover states.
- Loading states: use a subtle pulse animation on skeleton placeholders, never a spinner.

---

## File Structure

```
fracture-dashboard/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.js
├── src/
│   ├── main.tsx                    # ReactDOM.createRoot entry
│   ├── App.tsx                     # Router + layout shell
│   ├── api/
│   │   ├── types.ts                # All TypeScript interfaces (mirrors Rust serde structs)
│   │   ├── client.ts               # Typed fetch wrapper for REST endpoints
│   │   └── sse.ts                  # SSE connection helpers
│   ├── hooks/
│   │   ├── useCluster.ts           # GET /v1/cluster polling
│   │   ├── useScheduler.ts         # GET /v1/scheduler polling
│   │   ├── useMetricsStream.ts     # SSE /v1/metrics/stream → rolling window
│   │   ├── useInference.ts         # POST /v1/chat/completions with SSE streaming
│   │   └── useRequests.ts          # GET /v1/requests polling
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Shell.tsx           # Sidebar + main content area
│   │   │   ├── Sidebar.tsx         # Navigation links
│   │   │   └── StatusBar.tsx       # Bottom bar with connection status
│   │   ├── cluster/
│   │   │   ├── PipelineView.tsx    # Horizontal worker node chain with arrows
│   │   │   ├── WorkerCard.tsx      # Single worker: GPU, VRAM bar, layers, heartbeat
│   │   │   ├── ModelInfo.tsx       # Model metadata card
│   │   │   └── MetricsSummary.tsx  # Live sparklines row
│   │   ├── playground/
│   │   │   ├── ChatView.tsx        # Multi-turn chat with streaming
│   │   │   ├── MessageBubble.tsx   # Single message (user or assistant)
│   │   │   ├── ParamControls.tsx   # Sliders and inputs for sampling params
│   │   │   └── GenerationStats.tsx # Tokens/sec, TTFT, total tokens
│   │   ├── scheduler/
│   │   │   ├── QueueOverview.tsx   # Decode vs prefill queue counts
│   │   │   ├── CachePool.tsx       # KV block pool visualization
│   │   │   └── SequenceTable.tsx   # Active sequences table
│   │   └── shared/
│   │       ├── Sparkline.tsx       # Tiny inline line chart
│   │       ├── ProgressBar.tsx     # Horizontal bar with fill %
│   │       ├── StatusDot.tsx       # Colored circle indicator
│   │       └── Skeleton.tsx        # Loading placeholder
│   ├── pages/
│   │   ├── ClusterPage.tsx
│   │   ├── PlaygroundPage.tsx
│   │   └── SchedulerPage.tsx
│   └── lib/
│       ├── formatters.ts           # formatTokens, formatDuration, formatBytes, etc.
│       └── constants.ts            # API base URL, polling intervals, buffer sizes
```

---

## API Contracts

All endpoints are served by the Fracture backend. Base URL is configurable via `VITE_API_URL` env var, defaulting to `http://localhost:8080`.

### Existing Endpoints (already implemented in Rust)

#### `GET /health`

```typescript
// Response
interface HealthResponse {
  status: "ok";
}
```

#### `GET /v1/models`

```typescript
interface ModelsResponse {
  object: "list";
  data: Array<{
    id: string;
    object: "model";
    owned_by: string;
  }>;
}
```

#### `POST /v1/chat/completions`

```typescript
// Request
interface ChatCompletionRequest {
  model?: string;
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string;
  }>;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
  stream?: boolean;
  stop?: string[];
}

// Non-streaming response
interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: "assistant";
      content: string;
    };
    finish_reason: "stop" | "length";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// Streaming: each SSE event is `data: <json>\n\n`
// where json is:
interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant";
      content?: string;
    };
    finish_reason: "stop" | "length" | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
// Final event: `data: [DONE]\n\n`
```

#### `POST /v1/completions`

```typescript
// Request
interface CompletionRequest {
  model?: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
  stream?: boolean;
  stop?: string[];
}

// Response follows same pattern as chat but with `text` instead of `message`
```

### New Endpoints (to be implemented in Rust)

These endpoints need to be added to the Rust backend. They expose data that already exists in memory — no new computation required.

#### `GET /v1/cluster`

Returns cluster topology and worker status. In single-node mode, returns a single worker representing the local GPU.

```typescript
interface ClusterResponse {
  mode: "standalone" | "distributed";
  num_workers: number;
  workers: WorkerInfo[];
  scheduling_mode: "auto" | "equal_split" | "manual";
  model: {
    name: string;
    parameters: string;
    layers: number;
    context_length: number;
    dtype: string;
  };
}

interface WorkerInfo {
  id: number;
  role: "head" | "middle" | "tail" | "standalone";
  address: string;
  gpu: string;
  vram_total_mb: number;
  vram_used_mb: number;
  layers: [number, number];  // [start, end] inclusive
  status: "active" | "dead" | "calibrating";
  last_heartbeat_ms: number;
  calibration_ms_per_layer: number;
}
```

**Rust implementation notes:**
- In `fracture-coordinator`: serialize from `PeerRegistry` entries + `SchedulerResult`
- In `fracture-server` (standalone mode): return single worker with info from the CUDA backend's device query
- Worker `status` is derived from heartbeat age: active if < 10s, dead otherwise
- `calibration_ms_per_layer` comes from the calibration step that already runs at worker registration

#### `GET /v1/scheduler`

Returns batch scheduler state. Only meaningful when the server is actively handling requests.

```typescript
interface SchedulerResponse {
  active_sequences: number;
  max_sequences: number;
  decode_queue: number;
  prefill_queue: number;
  prefill_chunk_size: number;
  kv_cache: {
    block_size: number;
    total_blocks: number;
    allocated_blocks: number;
    free_blocks: number;
    utilization: number;  // 0.0 to 1.0
  };
  sequences: SequenceInfo[];
}

interface SequenceInfo {
  id: string;
  state: "prefilling" | "decoding" | "completed";
  tokens_generated: number;
  max_tokens: number;
  prefill_tokens: number;
  cache_blocks_held: number;
  started_at: string;  // ISO 8601
}
```

**Rust implementation notes:**
- Serialize from `BatchScheduler` state — it already tracks all active sequences, their states, and the block pool
- `kv_cache` info comes from the `BlockPool` (paged KV cache manager)
- This is a snapshot — sequence list may change between requests

#### `GET /v1/metrics/stream`

SSE endpoint that pushes a metrics snapshot every second.

```typescript
// Each SSE event: `data: <json>\n\n`
interface MetricsEvent {
  timestamp: string;  // ISO 8601
  throughput_tokens_per_sec: number;
  active_requests: number;
  avg_time_to_first_token_ms: number;
  avg_inter_token_latency_ms: number;
  kv_cache_utilization: number;
  worker_heartbeats: number[];  // ms since last heartbeat per worker
}
```

**Rust implementation notes:**
- New `MetricsCollector` struct in `fracture-server` that aggregates stats from completed requests
- Throughput: track tokens generated in a sliding 10s window
- TTFT and ITL: exponential moving average from recent completions
- Push via the same SSE mechanism used for streaming completions (axum's `Sse` extractor)
- Send empty/zero metrics if no recent requests — the stream should never stop

#### `GET /v1/requests`

Returns recent completed request metadata.

```typescript
interface RequestsResponse {
  requests: RequestRecord[];
  total: number;
  page: number;
  per_page: number;
}

interface RequestRecord {
  id: string;
  type: "chat" | "completion";
  status: "completed" | "cancelled" | "error";
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  time_to_first_token_ms: number;
  total_duration_ms: number;
  tokens_per_second: number;
  finish_reason: "stop" | "length" | "cancelled" | "error";
  temperature: number;
  created_at: string;  // ISO 8601
}
```

**Rust implementation notes:**
- Add a `RequestLog` (bounded `VecDeque<RequestRecord>`, cap at 1000) to the server's shared state
- Log each completed request after the response stream finishes
- Support `?page=N&per_page=M` query params
- Timing data: record timestamps at request start, first token, and completion

---

## Frontend Implementation Details

### Vite Config

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
```

This means during development, the frontend just calls `/v1/cluster` etc. and Vite proxies to the Rust backend. No CORS issues in dev. In production, the dashboard is served from the same origin.

### API Client (`src/api/client.ts`)

Typed wrapper around fetch. Every endpoint gets its own function.

```typescript
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
  models: () => fetchJSON<ModelsResponse>('/v1/models'),
  cluster: () => fetchJSON<ClusterResponse>('/v1/cluster'),
  scheduler: () => fetchJSON<SchedulerResponse>('/v1/scheduler'),
  requests: (page = 1, perPage = 50) =>
    fetchJSON<RequestsResponse>(`/v1/requests?page=${page}&per_page=${perPage}`),
};
```

### SSE Streaming for Chat (`src/api/sse.ts`)

The chat streaming uses POST with SSE response, which `EventSource` doesn't support (it's GET-only). Use `fetch` with a `ReadableStream` reader instead.

```typescript
export async function streamChat(
  request: ChatCompletionRequest,
  onChunk: (chunk: ChatCompletionChunk) => void,
  onDone: () => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, stream: true }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream error: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed === '') continue;
      if (trimmed === 'data: [DONE]') {
        onDone();
        return;
      }
      if (trimmed.startsWith('data: ')) {
        const json = trimmed.slice(6);
        try {
          const chunk: ChatCompletionChunk = JSON.parse(json);
          onChunk(chunk);
        } catch {
          // Skip malformed chunks
        }
      }
    }
  }
  onDone();
}
```

### SSE for Metrics Stream (`src/hooks/useMetricsStream.ts`)

The metrics endpoint is GET-based SSE, so native `EventSource` works fine.

```typescript
export function useMetricsStream(bufferSize = 60) {
  const [metrics, setMetrics] = useState<MetricsEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const es = new EventSource('/v1/metrics/stream');

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
```

### Polling Hooks Pattern

For endpoints that aren't SSE, use a simple polling hook:

```typescript
export function useCluster(intervalMs = 5000) {
  const [data, setData] = useState<ClusterResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const result = await api.cluster();
        if (active) {
          setData(result);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setLoading(false);
        }
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { data, error, loading };
}
```

Use same pattern for `useScheduler` and `useRequests`.

---

## Page Implementations

### 1. Cluster Page (landing page, route: `/`)

**Layout:** Full-width. Pipeline visualization at top spanning the page. Below it, a row of metric summary cards. Below that, model info.

**PipelineView component:**
- Render workers as cards arranged horizontally with arrow connectors between them
- Each worker card shows: GPU name, layer range (e.g., "Layers 0–15"), VRAM usage bar, heartbeat status dot, calibration speed
- Arrow between cards: simple SVG or CSS border arrow pointing left to right
- Handle 1 worker (standalone), 2 workers, or N workers dynamically
- Worker card border color reflects status: emerald for active, red for dead, amber for calibrating

**MetricsSummary component:**
- Row of 3-4 stat cards, each with: current value (large mono text), label (small muted text), sparkline (last 60s)
- Metrics: Throughput (tok/s), Active Requests, TTFT (ms), Cache Utilization (%)
- Data comes from `useMetricsStream` hook

**ModelInfo component:**
- Simple card with: model name, parameter count, dtype, context length, total layers
- Data comes from `useCluster` hook's `model` field

### 2. Playground Page (route: `/playground`)

**Layout:** Two-column on large screens. Left: chat area (flexible width). Right: parameter controls + generation stats (fixed ~320px).

**ChatView component:**
- Message list with user and assistant bubbles
- User messages: right-aligned, `bg-blue-400/10` background
- Assistant messages: left-aligned, `bg-gray-800` background
- Input area at bottom: textarea (not input) with send button, supports Enter to send and Shift+Enter for newline
- While streaming: show assistant message growing token by token, with a blinking cursor at the end
- Abort button appears during generation (wired to AbortController)
- After generation: show stats inline below the assistant message

**ParamControls component:**
- Temperature: range slider 0.0–2.0, step 0.1, default 0.7
- Top-P: range slider 0.0–1.0, step 0.05, default 1.0
- Top-K: number input, default 0 (disabled)
- Max Tokens: number input, default 256
- Seed: number input, empty = random
- Each control: label, current value display, input element

**GenerationStats component:**
- Shown during and after generation
- Time to first token (ms)
- Tokens per second
- Total tokens (prompt + completion)
- Finish reason
- Use `font-mono` for all values

### 3. Scheduler Page (route: `/scheduler`)

**Layout:** Top section: overview stats + KV cache visualization. Bottom section: active sequences table.

**QueueOverview component:**
- Three stat cards: Active Sequences (N / max), Decode Queue count, Prefill Queue count
- Prefill chunk size displayed as secondary info

**CachePool component:**
- Visual representation of the KV block pool
- Horizontal bar: green fill for allocated, dark for free, with percentage label
- Color transitions: emerald (<60%), amber (60-85%), red (>85%)
- Below the bar: text stats — "847 / 2048 blocks (41.3%)"
- Block size info: "16 tokens per block"

**SequenceTable component:**
- Table columns: ID (mono, truncated), State (colored badge), Tokens (generated / max), Prefill Tokens, Cache Blocks, Duration
- State badges: "prefilling" = blue, "decoding" = emerald, "completed" = gray
- Sort by any column (client-side sort is fine)
- Auto-refresh via `useScheduler` hook (poll every 2s on this page)

---

## Routing

```typescript
// App.tsx
<Routes>
  <Route element={<Shell />}>
    <Route index element={<ClusterPage />} />
    <Route path="playground" element={<PlaygroundPage />} />
    <Route path="scheduler" element={<SchedulerPage />} />
  </Route>
</Routes>
```

**Sidebar navigation items:**
1. Cluster — icon: `Activity` or `Server`
2. Playground — icon: `MessageSquare`
3. Scheduler — icon: `Layers`

Active route gets highlighted state. Sidebar is narrow (64px icons-only or 200px with labels — your call, icons-only is cleaner for a dashboard).

---

## Build Priority

This is the order to implement. Stop at any point and you have a working, presentable dashboard.

### Phase 1: Foundation
1. Scaffold Vite + React + TS + Tailwind project
2. Implement `api/types.ts` with all interfaces
3. Implement `api/client.ts` and `api/sse.ts`
4. Implement `Shell`, `Sidebar` layout components
5. Set up React Router with three pages (can be placeholder content)

### Phase 2: Cluster Overview
1. Implement `useCluster` and `useMetricsStream` hooks
2. Build `WorkerCard` component
3. Build `PipelineView` composing worker cards with connectors
4. Build `ModelInfo` card
5. Build `Sparkline` shared component
6. Build `MetricsSummary` with live sparklines
7. Assemble `ClusterPage`

### Phase 3: Inference Playground
1. Implement `useInference` hook (manages messages, streaming, abort)
2. Build `MessageBubble` component
3. Build `ChatView` with streaming display
4. Build `ParamControls`
5. Build `GenerationStats`
6. Assemble `PlaygroundPage`

### Phase 4: Scheduler View
1. Implement `useScheduler` hook
2. Build `QueueOverview` stat cards
3. Build `CachePool` visualization
4. Build `SequenceTable` with sorting
5. Assemble `SchedulerPage`

---

## Backend Implementation Order

Do this first or in parallel with Phase 1.

1. **CORS middleware** — Add `tower-http`'s `CorsLayer` to the axum router. Allow all origins in dev.
2. **`GET /v1/cluster`** — Serialize `PeerRegistry` + model config. For standalone mode, construct a single-worker response from the backend's device info.
3. **`GET /v1/metrics/stream`** — Create `MetricsCollector` that tracks a sliding window of request stats. Spawn a tokio task that sends SSE events every second.
4. **`GET /v1/scheduler`** — Serialize `BatchScheduler` snapshot. Return empty sequences list if no active requests.
5. **`GET /v1/requests`** — Add `RequestLog` (bounded VecDeque) to shared server state. Log each completed request. Support pagination query params.
6. **Static file serving** — Use `tower-http`'s `ServeDir` to serve `fracture-dashboard/dist/` at `/dashboard`. This is last because it's not needed during dev (Vite proxy handles it).

---

## Error Handling

- All API calls should handle network errors gracefully
- Show a connection status indicator in the sidebar or status bar (green dot = connected, red = disconnected)
- If `useMetricsStream` disconnects, show a "Reconnecting..." state and attempt reconnection after 3s
- If cluster endpoint returns an error, show an error card instead of the pipeline view, not a blank page
- In the playground, if the stream errors mid-generation, show what was received so far + an error indicator

---

## Things to NOT Do

- Don't add authentication. This is a local/internal tool.
- Don't add a database. All state is in-memory.
- Don't add WebSocket support. SSE is already implemented in the backend and is sufficient.
- Don't use a CSS-in-JS library. Tailwind only.
- Don't add dark/light theme toggle. Dark only.
- Don't over-abstract. If a component is used once, it doesn't need to be in `shared/`.
- Don't add animations/transitions beyond basic hover states and the streaming cursor blink.