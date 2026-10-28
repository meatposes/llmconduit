# llmconduit

Multi-backend OpenAI-compatible aggregating reverse proxy. Run multiple LLM inference servers and expose them all through a single endpoint with unified auth, stable role aliases, and automatic model discovery.

## How it works

```mermaid
flowchart TD
    Client["Client\nOpenAI-compatible\n(any tool / SDK)"]

    subgraph llmconduit ["llmconduit  (FastAPI)"]
        Auth["Auth\nPROXY_API_KEY"]
        Router["Request Router\nresolves model → backend\nrewrites model field\ninjects backend auth"]
        Registry["BackendRegistry\n• routing table\n• alias resolution\n• health state"]
        Scanner["Background Scanner\npolls /v1/models\nevery N seconds"]
        AdminUI["Admin UI + API\n/admin/ui\n/admin/alias/:name\n/admin/rescan"]
    end

    BackendA["Backend A\nllama.cpp / vLLM\nOllama / etc."]
    BackendB["Backend B\nany OpenAI-compat\nAPI"]

    Client -->|"Bearer PROXY_API_KEY\nmodel: PRIMARY"| Auth
    Auth --> Router
    Router -->|"resolve PRIMARY\n→ model-id@backend-a"| Registry
    Registry --> Router
    Router -->|"POST /v1/chat/completions\nmodel: model-id (rewritten)"| BackendA
    Router -.->|"or route to"| BackendB
    BackendA -->|"response / stream"| Client
    Scanner -->|"GET /v1/models"| BackendA
    Scanner -->|"GET /v1/models"| BackendB
    Scanner -->|"update routing table"| Registry
    AdminUI -->|"override alias target"| Registry
```

**Request flow:**
1. Client authenticates with `PROXY_API_KEY` and sends `model: "model-id@backend-name"` or a role alias (`PRIMARY`, `AUXILIARY`, …)
2. llmconduit resolves the alias → qualified model, strips the `@backend-name` suffix, rewrites the request body, injects backend auth if configured, and forwards the request
3. Streaming and non-streaming responses are passed through transparently
4. If a model isn't in the routing table, llmconduit rescans that backend before returning 404

## Features

- **Single endpoint** — one URL, one API key for all downstream clients
- **Role aliases** — stable names (`PRIMARY`, `CODING`, …) that you retarget at runtime without touching any client config
- **Auto-discovery** — polls each backend's `/v1/models` on a configurable interval
- **Miss-triggered rescan** — on a routing miss, rescans the target backend before failing
- **Per-backend auth** — forward caller auth, strip it, or inject a backend-specific key
- **Admin UI** — browser-based alias switcher at `/admin/ui?token=<ADMIN_TOKEN>`
- **Admin API** — programmatic alias overrides, rescan, reload via `/admin/*` endpoints
- **Streaming** — full SSE / chunked-transfer passthrough
- **Prometheus metrics** — one `/metrics` for everything: requests, latency, time to first token and tokens per model and backend, plus every backend's own metrics tagged with its name
- **Docker-native** — single container, config mounted as a volume

## Quickstart

```bash
# 1. REQUIRED — copy the template and set your secrets before anything else
cp .env.template .env
#   open .env and set PROXY_API_KEY to any strong secret string
#   optionally set ADMIN_TOKEN to enable the /admin/* endpoints

# 2. edit config.yaml — uncomment and fill in your backends and aliases

# 3. bring it up
docker compose up -d
```

> **Note:** The container will refuse to start if `PROXY_API_KEY` is not set in `.env`.

The proxy listens on port `8000` by default (`PROXY_PORT` to override).

## Configuration

### `config.yaml`

```yaml
backends:
  - url: http://my-llama-server:8080
    # no api_key — caller's Authorization header forwarded as-is

  - url: http://another-server:11434
    api_key: ANOTHER_SERVER_KEY   # env var name; value set in .env

  - url: http://192.168.1.50:8000
    alias: openarc-node           # models show as model-id@openarc-node

aliases:
  PRIMARY: llama3.1-8b@my-llama-server
  CODING:  deepseek-coder@another-server
```

- **Backend name** is derived from the hostname automatically, or set explicitly with `alias`
- **Models** are exposed as `model-id@backend-name`
- **Aliases** are case-insensitive; a role only appears in `/v1/models` while its target backend is healthy
- Change `config.yaml` and call `POST /admin/reload` (or `docker compose restart`) to apply

#### Backend `alias`

`alias` replaces the hostname-derived name in every model id, so `mistral-7b@192.168.1.50`
becomes `mistral-7b@openarc-node`.

It is **required** when two backends share a hostname — several vLLM instances on
different ports of one machine, for example. Without it both are named after the same
host, their model ids collide, and only one of them stays reachable. An alias must be
unique across backends and must not contain `@`; an invalid alias is ignored (logged
as an error) and the hostname is used instead.

#### Keeping your real backends out of git

`config.yaml` is tracked, so edits to it conflict on every `git pull` and any internal
hostnames end up in the repo. Copy it instead:

```bash
cp config.yaml config.local.yaml     # gitignored
echo "CONFIG_FILE=config.local.yaml" >> .env
docker compose up -d
```

### Environment variables

All variables are set in `.env` (copy from `.env.template`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `PROXY_API_KEY` | **Yes** | — | Shared key all clients use to authenticate |
| `ADMIN_TOKEN` | No | — | Enables `/admin/*` endpoints; omit to disable |
| `CONFIG_PATH` | No | `/etc/llmconduit/config.yaml` | Path to config file (override for local runs) |
| `REFRESH_INTERVAL_SECONDS` | No | `60` | How often to poll backends for model lists |
| `MISS_RETRY_ATTEMPTS` | No | `3` | Rescans before returning 404 on a routing miss |
| `MISS_RETRY_DELAY_SECONDS` | No | `2` | Delay between miss retry rescans |
| `BACKEND_SCAN_TIMEOUT` | No | `10` | Timeout for `/v1/models` scan requests |
| `REQUEST_TIMEOUT` | No | `600` | Timeout for proxied inference requests |
| `REQUEST_CONNECT_TIMEOUT` | No | `10` | Connection establishment timeout |
| `LOG_LEVEL` | No | `info` | `debug`, `info`, `warning`, `error` |
| `PROXY_PORT` | No | `8000` | Port the proxy listens on |
| `METRICS_TOKEN` | No | — | If set, `/metrics` requires `Authorization: Bearer <METRICS_TOKEN>`; otherwise it is open |
| `BACKEND_METRICS` | No | `true` | Include each backend's own `/metrics` in llmconduit's `/metrics` |
| `BACKEND_METRICS_TIMEOUT` | No | `5` | Seconds to wait for one backend's `/metrics` before leaving it out of that scrape |
| `STREAM_USAGE` | No | `true` | Ask backends for token usage on streamed chat/completions so it can be counted (see Metrics) |

Per-backend API keys are also set in `.env` and referenced by name in `config.yaml`:
```
EXTERNAL_API_KEY=sk-...
```

## API

All inference endpoints require `Authorization: Bearer <PROXY_API_KEY>`.

### Inference

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List all available models and aliases |
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/embeddings` | Embeddings |
| `POST` | `/v1/audio/*` | Speech, transcriptions, translations and backend extras; JSON or multipart, response streamed through as-is |
| `POST` | `/v1/rerank` | Rerank |
| `*` | `/{path}` | Catch-all: routes to first healthy backend |

### Health

```
GET /health
```
Unauthenticated. Returns backend counts and model count.

### Metrics

| Method | Path | Description |
|---|---|---|
| `GET` | `/metrics` | Everything, text or OpenMetrics by `Accept`: llmconduit's own metrics plus every healthy backend's `/metrics` |

Open unless `METRICS_TOKEN` is set. The inference key never works here, and the metrics token never works for inference.

On each scrape llmconduit reads every healthy backend's `/metrics` in parallel (with the backend's own key, if it has one) and adds a `backend` label to every sample, so vLLM's KV cache, prefix-cache and queue numbers arrive next to the gateway's. A backend with no `/metrics` (404) is left out; one that fails or takes longer than `BACKEND_METRICS_TIMEOUT` is skipped for that scrape and shows `llmconduit_backend_metrics_up 0`. A label already called `backend` becomes `exported_backend`. `*_created` series are dropped.

What llmconduit measures, labelled by `model` and `backend` (names follow the OpenTelemetry GenAI metrics):

| Metric | Type | Labels |
|---|---|---|
| `llmconduit_requests_total` | counter | `endpoint`, `model`, `backend`, `alias`, `status` |
| `llmconduit_request_duration_seconds` | histogram | `endpoint`, `model`, `backend` |
| `llmconduit_time_to_first_token_seconds` | histogram | streamed chat/completions |
| `llmconduit_time_per_output_token_seconds` | histogram | streamed chat/completions with usage |
| `llmconduit_tokens_total` | counter | `type` = `input` or `output`, from response usage |
| `llmconduit_requests_in_flight` | gauge | |
| `llmconduit_backend_errors_total` | counter | `backend`, `kind` |
| `llmconduit_backend_up` | gauge | `backend`, from the last model scan |
| `llmconduit_alias_info` | gauge | `alias`, `model`, `backend`, `source` — where each alias points |
| `llmconduit_backend_metrics_up` | gauge | `backend` — 1 if its `/metrics` was read on this scrape |
| `llmconduit_backend_metrics_scrape_seconds` | gauge | `backend` |

- `status` is the backend's HTTP status, `502` when the backend could not be reached or dropped mid-request, or `499` when the client hung up mid-stream. Requests rejected before routing (unknown model, bad body) are counted with empty `model` and `backend`.
- Token counts on streams need the backend's final usage chunk. With `STREAM_USAGE=true`, llmconduit adds `stream_options.include_usage` when the caller did not, and drops that extra chunk before it reaches the caller. Chunks may carry `"usage": null`, as the OpenAI spec allows.

To follow an alias whichever backend it points at, join on `llmconduit_alias_info`:

```promql
# gateway numbers
sum by (model) (rate(llmconduit_tokens_total{type="output"}[5m])
  * on (model, backend) group_left (alias) llmconduit_alias_info{alias="PRIMARY"})
# a backend's own numbers
vllm:kv_cache_usage_perc * on (backend) group_left (alias) llmconduit_alias_info{alias="PRIMARY"}
```

Prometheus config:

```yaml
scrape_configs:
  - job_name: llmconduit
    static_configs:
      - targets: ['llmconduit-host:8000']
    # authorization: {credentials_file: /etc/prometheus/llmconduit-metrics-token}   # if METRICS_TOKEN is set
```

### Admin (requires `ADMIN_TOKEN`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/status` | Full routing table and backend health |
| `POST` | `/admin/rescan` | Rescan all backends |
| `POST` | `/admin/rescan/{backend}` | Rescan one backend |
| `POST` | `/admin/reload` | Re-read `config.yaml` + rescan all |
| `GET` | `/admin/aliases` | List current alias state |
| `POST` | `/admin/alias/{name}` | Override alias target: `{"target": "model@backend"}` |
| `DELETE` | `/admin/alias/{name}` | Clear override (revert to config default) |
| `GET` | `/admin/ui` | Browser-based alias switcher (`?token=<ADMIN_TOKEN>`) |

## Docker

`config.yaml` is bind-mounted read-only into the container at `/etc/llmconduit/config.yaml`.

By default, `docker-compose.yml` uses Docker's default bridge network. llmconduit reaches backends by IP or by hostname if they're on the same compose network.

### Connecting to backends on an external Docker network

If your LLM backends live in a separate compose stack on a named network, attach llmconduit to it with an override file:

```yaml
# docker-compose.override.yml
services:
  llmconduit:
    networks:
      - ai-network

networks:
  ai-network:
    external: true
```

Docker Compose merges this with `docker-compose.yml` automatically on `docker compose up`. You can name the network whatever matches your existing stack.

### Running without Docker

```bash
pip install -r app/requirements.txt
CONFIG_PATH=./config.yaml PROXY_API_KEY=your-key uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
