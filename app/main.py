"""
llmconduit
Multi-backend OpenAI-compatible aggregating reverse proxy.
"""

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
# Context-window keys as they appear across OpenAI-compatible backends: vLLM
# reports max_model_len, llama.cpp n_ctx / n_ctx_train, others context_length.
# Clients (e.g. Hermes) probe for these same names, so dropping them here makes
# every model look like the client's minimum-window fallback.
_CONTEXT_LENGTH_KEYS = (
    "max_model_len", "context_length", "context_window", "max_context_length",
    "max_position_embeddings", "max_input_tokens", "max_sequence_length",
    "max_seq_len", "n_ctx_train", "n_ctx", "ctx_size",
)


def _extract_context_length(entry: dict) -> Optional[int]:
    """Return the context window advertised by a /v1/models entry, if any."""
    if not isinstance(entry, dict):
        return None
    for key in _CONTEXT_LENGTH_KEYS:
        v = entry.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v > 0:
            return v
        if isinstance(v, str) and v.isdigit() and int(v) > 0:
            return int(v)
    return _extract_context_length(entry.get("meta")) if isinstance(entry.get("meta"), dict) else None


PROXY_API_KEY            = os.environ.get("PROXY_API_KEY", "")
ADMIN_TOKEN              = os.environ.get("ADMIN_TOKEN", "")
REFRESH_INTERVAL         = int(os.environ.get("REFRESH_INTERVAL_SECONDS", "60"))
MISS_RETRY_ATTEMPTS      = int(os.environ.get("MISS_RETRY_ATTEMPTS", "3"))
MISS_RETRY_DELAY         = float(os.environ.get("MISS_RETRY_DELAY_SECONDS", "2"))
BACKEND_SCAN_TIMEOUT     = float(os.environ.get("BACKEND_SCAN_TIMEOUT", "10"))
REQUEST_TIMEOUT          = float(os.environ.get("REQUEST_TIMEOUT", "600"))
REQUEST_CONNECT_TIMEOUT  = float(os.environ.get("REQUEST_CONNECT_TIMEOUT", "10"))
LOG_LEVEL                = os.environ.get("LOG_LEVEL", "info").upper()
CONFIG_PATH              = os.environ.get("CONFIG_PATH", "/etc/llmconduit/config.yaml")

ADMIN_ENABLED = bool(ADMIN_TOKEN)

if not PROXY_API_KEY:
    raise RuntimeError("PROXY_API_KEY env var is required but not set")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("llmconduit")

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}

def resolve_backend_key(key_ref: Optional[str]) -> Optional[str]:
    """key_ref is an env var name. Resolve it to the actual value."""
    if not key_ref:
        return None
    val = os.environ.get(key_ref)
    if not val:
        log.warning("Backend api_key env var '%s' is not set", key_ref)
    return val

def backend_name_from_url(url: str) -> str:
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0]
    return url.split(":")[0]

def normalize_alias(url: str, alias: Optional[str]) -> Optional[str]:
    """Validate a configured backend alias. Returns the cleaned alias, or None
    if it is unset or unusable — in which case the hostname is used instead."""
    if alias is None:
        return None
    alias = str(alias).strip()
    if not alias:
        return None
    if "@" in alias:
        log.error(
            "Backend '%s': alias '%s' contains '@' — ignoring it and using the hostname. "
            "Models are exposed as '<model-id>@<backend-name>', so an '@' in the alias "
            "breaks model resolution.",
            url, alias,
        )
        return None
    return alias

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class BackendRegistry:
    def __init__(self):
        # { backend_url: { name, models, healthy, api_key_ref, alias } }
        self._backends: dict[str, dict] = {}
        # { qualified_model_id: backend_url }
        self._routing: dict[str, str] = {}
        # { ALIAS_NAME: qualified_model_id } — config-defined defaults, refreshed on every scan
        self._aliases: dict[str, str] = {}
        # { ALIAS_NAME: qualified_model_id } — admin-set, takes priority over _aliases.
        # In-memory only; survives scans/reloads, cleared on restart or when the role
        # is removed from config.yaml entirely.
        self._alias_overrides: dict[str, str] = {}
        # Coalescing locks per backend URL
        self._rescan_locks: dict[str, asyncio.Lock] = {}
        self._rw_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rescan_lock_for(self, url: str) -> asyncio.Lock:
        if url not in self._rescan_locks:
            self._rescan_locks[url] = asyncio.Lock()
        return self._rescan_locks[url]

    async def _fetch_models(self, client: httpx.AsyncClient, url: str, api_key: Optional[str]) -> list[dict]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = await client.get(
            f"{url}/v1/models",
            headers=headers,
            timeout=BACKEND_SCAN_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [m for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]

    async def _scan_one(self, client: httpx.AsyncClient, url: str, api_key_ref: Optional[str], alias: Optional[str] = None) -> dict:
        alias = normalize_alias(url, alias)
        bname = alias or backend_name_from_url(url)
        api_key = resolve_backend_key(api_key_ref)
        try:
            entries = await self._fetch_models(client, url, api_key)
            models = [m["id"] for m in entries]
            model_meta = {}
            for m in entries:
                ctx = _extract_context_length(m)
                if ctx:
                    model_meta[m["id"]] = ctx
            log.info("Backend '%s' healthy — models: %s (context: %s)", bname, models, model_meta or "not advertised")
            return {"name": bname, "models": models, "model_meta": model_meta,
                    "healthy": True, "api_key_ref": api_key_ref, "alias": alias}
        except Exception as exc:
            log.warning("Backend '%s' unreachable: %s", bname, exc)
            return {"name": bname, "models": [], "model_meta": {},
                    "healthy": False, "api_key_ref": api_key_ref, "alias": alias}

    async def _rebuild_routing(self, new_backends: dict[str, dict], new_aliases: dict[str, str]):
        new_routing: dict[str, str] = {}
        seen_names: dict[str, str] = {}
        for url, info in new_backends.items():
            first_url = seen_names.setdefault(info["name"], url)
            if first_url != url:
                log.error(
                    "Backends '%s' and '%s' both resolve to the name '%s' — their model ids "
                    "collide and targeted rescans only reach the first. Give each a unique 'alias'.",
                    first_url, url, info["name"],
                )
        for url, info in new_backends.items():
            if info["healthy"]:
                for mid in info["models"]:
                    qualified = f"{mid}@{info['name']}"
                    new_routing[qualified] = url
        async with self._rw_lock:
            self._backends = new_backends
            self._routing = new_routing
            self._aliases = new_aliases
            # Role removed from config.yaml entirely — drop any override riding on it too.
            for name in list(self._alias_overrides):
                if name not in new_aliases:
                    del self._alias_overrides[name]
        log.info(
            "Routing table: %d models across %d backends (%d healthy)",
            len(new_routing),
            len(new_backends),
            sum(1 for i in new_backends.values() if i["healthy"]),
        )
        for name, target in new_aliases.items():
            if target not in new_routing:
                log.warning("Alias '%s' target '%s' is not currently resolvable", name, target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def full_scan(self, cfg: Optional[dict] = None):
        """Scan all configured backends."""
        if cfg is None:
            cfg = load_config()
        entries = cfg.get("backends", [])
        aliases = {str(k).upper(): v for k, v in (cfg.get("aliases") or {}).items()}
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *[self._scan_one(client, e["url"].rstrip("/"), e.get("api_key"), e.get("alias")) for e in entries],
                return_exceptions=False,
            )
        new_backends = {
            e["url"].rstrip("/"): result
            for e, result in zip(entries, results)
        }
        await self._rebuild_routing(new_backends, aliases)

    async def scan_backend(self, backend_url: str):
        """Targeted rescan of one backend. Coalesces concurrent calls."""
        backend_url = backend_url.rstrip("/")
        lock = self._rescan_lock_for(backend_url)
        async with lock:
            # Check if another coroutine already refreshed while we waited
            cfg = load_config()
            entry = next((e for e in cfg.get("backends", []) if e["url"].rstrip("/") == backend_url), None)
            if not entry:
                log.warning("Targeted rescan: backend '%s' not in config", backend_url)
                return
            async with httpx.AsyncClient() as client:
                info = await self._scan_one(client, backend_url, entry.get("api_key"), entry.get("alias"))
            async with self._rw_lock:
                self._backends[backend_url] = info
                # Remove old routes for this backend, add new ones
                self._routing = {
                    k: v for k, v in self._routing.items() if v != backend_url
                }
                if info["healthy"]:
                    for mid in info["models"]:
                        qualified = f"{mid}@{info['name']}"
                        self._routing[qualified] = backend_url

    async def reload_config(self):
        """Re-read config and full scan. Picks up new/removed backends."""
        log.info("Reloading config and rescanning all backends")
        await self.full_scan(load_config())

    def resolve(self, qualified: str) -> Optional[str]:
        return self._routing.get(qualified)

    def backend_url_for_name(self, name: str) -> Optional[str]:
        for url, info in self._backends.items():
            if info["name"] == name:
                return url
        return None

    def backend_api_key(self, backend_url: str) -> Optional[str]:
        info = self._backends.get(backend_url, {})
        return resolve_backend_key(info.get("api_key_ref"))

    def resolve_alias(self, name: str) -> Optional[str]:
        """Resolve a role alias (e.g. PRIMARY) to its qualified model@backend target."""
        key = name.upper()
        if key in self._alias_overrides:
            return self._alias_overrides[key]
        return self._aliases.get(key)

    def set_alias_override(self, name: str, target: str) -> None:
        key = name.upper()
        if key not in self._aliases:
            raise ValueError(f"Unknown alias '{name}' — not defined in config.yaml")
        if not self.resolve(target):
            raise ValueError(f"Target '{target}' is not a currently resolvable model")
        self._alias_overrides[key] = target

    def clear_alias_override(self, name: str) -> bool:
        return self._alias_overrides.pop(name.upper(), None) is not None

    def alias_status(self) -> dict[str, dict]:
        out = {}
        for name, default_target in self._aliases.items():
            target = self._alias_overrides.get(name, default_target)
            out[name] = {
                "target": target,
                "source": "override" if name in self._alias_overrides else "config",
                "resolved": self.resolve(target) is not None,
            }
        return out

    def all_models(self) -> list[dict]:
        now = int(time.time())
        seen = set()
        out = []
        for url, info in self._backends.items():
            if not info["healthy"]:
                continue
            for mid in info["models"]:
                qualified = f"{mid}@{info['name']}"
                if qualified not in seen:
                    seen.add(qualified)
                    entry = {
                        "id": qualified,
                        "object": "model",
                        "created": now,
                        "owned_by": info["name"],
                    }
                    ctx = info.get("model_meta", {}).get(mid)
                    if ctx:
                        entry["max_model_len"] = ctx
                        entry["context_length"] = ctx
                    out.append(entry)
        for name, default_target in self._aliases.items():
            target = self._alias_overrides.get(name, default_target)
            backend_url = self.resolve(target)
            if not backend_url:
                continue
            binfo = self._backends.get(backend_url, {})
            entry = {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": binfo.get("name", target.rsplit("@", 1)[-1]),
            }
            ctx = binfo.get("model_meta", {}).get(target.rsplit("@", 1)[0])
            if ctx:
                entry["max_model_len"] = ctx
                entry["context_length"] = ctx
            out.append(entry)
        return out

    def status(self) -> dict:
        return {
            "backends": {
                url: {
                    "name": info["name"],
                    "alias": info.get("alias"),
                    "healthy": info["healthy"],
                    "models": [f"{m}@{info['name']}" for m in info["models"]],
                    "has_api_key": bool(info.get("api_key_ref")),
                }
                for url, info in self._backends.items()
            },
            "routing_table": self._routing,
            "total_models": len(self._routing),
            "aliases": self.alias_status(),
        }


registry = BackendRegistry()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
security = HTTPBearer(auto_error=False)

async def require_proxy_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.credentials != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

async def require_admin_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    if not credentials or credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")

# ---------------------------------------------------------------------------
# Lifespan: startup scan + background refresh
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("llmconduit starting — scanning backends...")
    await registry.full_scan()
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

async def _refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        try:
            await registry.full_scan()
        except Exception as exc:
            log.error("Background refresh error: %s", exc)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="llmconduit", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------
def _make_timeout() -> httpx.Timeout:
    return httpx.Timeout(REQUEST_TIMEOUT, connect=REQUEST_CONNECT_TIMEOUT)

def _rewrite_model(body_bytes: bytes, raw_model_id: str) -> bytes:
    try:
        data = json.loads(body_bytes)
        data["model"] = raw_model_id
        return json.dumps(data).encode()
    except Exception:
        return body_bytes

def _inject_auth(headers: dict, backend_url: str, original_auth: Optional[str]) -> dict:
    backend_key = registry.backend_api_key(backend_url)
    if backend_key:
        headers["Authorization"] = f"Bearer {backend_key}"
    elif original_auth:
        headers["Authorization"] = original_auth
    else:
        headers.pop("Authorization", None)
    return headers

async def _proxy(
    request: Request,
    backend_url: str,
    rewritten_body: Optional[bytes] = None,
) -> Response:
    path = request.url.path
    query = request.url.query
    target = f"{backend_url}{path}"
    if query:
        target = f"{target}?{query}"

    original_auth = request.headers.get("Authorization")
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "authorization")
    }
    headers = _inject_auth(headers, backend_url, original_auth)

    body = rewritten_body if rewritten_body is not None else await request.body()
    is_stream = False
    try:
        is_stream = json.loads(body).get("stream", False)
    except Exception:
        pass

    if is_stream:
        async def _stream():
            try:
                async with httpx.AsyncClient(timeout=_make_timeout()) as client:
                    async with client.stream(
                        method=request.method,
                        url=target,
                        headers=headers,
                        content=body,
                    ) as resp:
                        async for chunk in resp.aiter_raw():
                            yield chunk
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                log.error("Stream error to %s: %s", backend_url, exc)
                yield b"data: {\"error\": \"backend connection lost\"}\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=_make_timeout()) as client:
                resp = await client.request(
                    method=request.method,
                    url=target,
                    headers=headers,
                    content=body,
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("content-encoding", "transfer-encoding")
                },
                media_type=resp.headers.get("content-type"),
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            log.error("Connection error to %s: %s", backend_url, exc)
            raise HTTPException(status_code=502, detail=f"Backend unreachable: {backend_url}")

async def _routed_proxy(request: Request) -> Response:
    """Resolve model → backend, with miss-triggered retry loop."""
    body = await request.body()
    backend_url, raw_model_id = await _resolve_model(_json_model(body))
    rewritten = _rewrite_model(body, raw_model_id)
    return await _proxy(request, backend_url, rewritten_body=rewritten)

def _json_model(body: bytes) -> str:
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    qualified = data.get("model", "") if isinstance(data, dict) else ""
    if not qualified:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    return qualified

async def _resolve_model(qualified: str) -> tuple[str, str]:
    """Model or role alias → (backend url, raw model id), rescanning the backend on a miss."""
    # Role alias (e.g. PRIMARY) → its current qualified target
    alias_target = registry.resolve_alias(qualified)
    if alias_target:
        qualified = alias_target

    # Derive backend name from qualified model string
    parts = qualified.rsplit("@", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be in 'model-id@backend-name' format, got: '{qualified}'"
        )
    backend_name = parts[1]

    # Attempt resolution with miss-triggered retry
    backend_url = registry.resolve(qualified)

    if not backend_url:
        target_url = registry.backend_url_for_name(backend_name)
        if not target_url:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Unknown backend '{backend_name}'",
                    "available_models": list(registry._routing.keys()),
                },
            )
        for attempt in range(1, MISS_RETRY_ATTEMPTS + 1):
            log.info(
                "Model '%s' not in routing table — rescanning '%s' (attempt %d/%d)",
                qualified, backend_name, attempt, MISS_RETRY_ATTEMPTS,
            )
            await registry.scan_backend(target_url)
            backend_url = registry.resolve(qualified)
            if backend_url:
                break
            if attempt < MISS_RETRY_ATTEMPTS:
                await asyncio.sleep(MISS_RETRY_DELAY)

    if not backend_url:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Model '{qualified}' not found after {MISS_RETRY_ATTEMPTS} rescan attempts",
                "available_models": list(registry._routing.keys()),
            },
        )

    log.info("Routing '%s' → %s", qualified, backend_url)
    return backend_url, qualified.rsplit("@", 1)[0]

# ---------------------------------------------------------------------------
# Audio and other non-chat endpoints
# ---------------------------------------------------------------------------
# These used to fall through to the catch-all, which sends everything to the
# first healthy backend whatever the model: a TTS request for an OpenArc model
# went to a vLLM backend and came back 404. They take JSON (speech, rerank) or
# multipart (transcriptions: the audio file plus a 'model' field), and answer
# with audio, text or JSON, sometimes streamed with the stream flag nested in a
# backend-specific block (OpenArc's openarc_tts.kokoro.stream). So: route by
# model like chat, and pass the response through as it comes, with the
# backend's own status and content type, without buffering it first.

def _multipart_boundary(content_type: str) -> Optional[bytes]:
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    return m.group(1).encode() if m else None

def _multipart_model(body: bytes, boundary: bytes) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """The 'model' field of a multipart body and the byte span of its value (for rewriting it)."""
    delim = b"--" + boundary
    pos = body.find(delim)
    while pos >= 0:
        head_start = pos + len(delim)
        if body[head_start:head_start + 2] == b"--":          # closing delimiter
            break
        head_end = body.find(b"\r\n\r\n", head_start)
        if head_end < 0:
            break
        value_start = head_end + 4
        nxt = body.find(b"\r\n" + delim, value_start)
        if nxt < 0:
            break
        headers = body[head_start:head_end].decode("latin-1")
        if re.search(r'^content-disposition:[^\r\n]*;\s*name="model"', headers, re.I | re.M):
            return body[value_start:nxt].decode("utf-8", "replace").strip(), (value_start, nxt)
        pos = nxt + 2
    return None, None

async def _routed_passthrough(request: Request) -> Response:
    """Route a JSON or multipart request by its model; stream the answer back untouched."""
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    if content_type.lower().startswith("multipart/form-data"):
        boundary = _multipart_boundary(content_type)
        qualified, span = _multipart_model(body, boundary) if boundary else (None, None)
        if not qualified:
            raise HTTPException(status_code=400, detail="Missing 'model' field")
        backend_url, raw_model_id = await _resolve_model(qualified)
        body = body[:span[0]] + raw_model_id.encode() + body[span[1]:]
    else:
        backend_url, raw_model_id = await _resolve_model(_json_model(body))
        body = _rewrite_model(body, raw_model_id)
    return await _passthrough(request, backend_url, body)

async def _passthrough(request: Request, backend_url: str, body: bytes) -> Response:
    target = f"{backend_url}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "authorization")
    }
    headers = _inject_auth(headers, backend_url, request.headers.get("Authorization"))

    # The upstream response is opened before the StreamingResponse is built, so
    # its status (and content type) are known in time; client and response
    # outlive this handler and are closed when the body is done.
    client = httpx.AsyncClient(timeout=_make_timeout())
    try:
        resp = await client.send(
            client.build_request(request.method, target, headers=headers, content=body), stream=True,
        )
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
        await client.aclose()
        log.error("Connection error to %s: %s", backend_url, exc)
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {backend_url}")
    if resp.status_code >= 400:
        log.warning("Backend %s returned %d for %s", backend_url, resp.status_code, request.url.path)

    async def _relay():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    passed = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length", "connection")
    }
    passed["X-Accel-Buffering"] = "no"
    return StreamingResponse(_relay(), status_code=resp.status_code, headers=passed)

# ---------------------------------------------------------------------------
# Inference routes
# ---------------------------------------------------------------------------
@app.get("/v1/models", dependencies=[Depends(require_proxy_auth)])
async def list_models():
    return {"object": "list", "data": registry.all_models()}

@app.post("/v1/chat/completions", dependencies=[Depends(require_proxy_auth)])
async def chat_completions(request: Request):
    return await _routed_proxy(request)

@app.post("/v1/completions", dependencies=[Depends(require_proxy_auth)])
async def completions(request: Request):
    return await _routed_proxy(request)

@app.post("/v1/embeddings", dependencies=[Depends(require_proxy_auth)])
async def embeddings(request: Request):
    return await _routed_proxy(request)

@app.post("/v1/audio/{endpoint:path}", dependencies=[Depends(require_proxy_auth)])
async def audio(request: Request, endpoint: str):
    """speech, transcriptions, translations, and backend extras (OpenArc's diarization, voiceprint)."""
    return await _routed_passthrough(request)

@app.post("/v1/rerank", dependencies=[Depends(require_proxy_auth)])
async def rerank(request: Request):
    return await _routed_passthrough(request)

@app.get("/health")
async def health():
    """Unauthenticated — used by Docker healthcheck."""
    alive = sum(1 for i in registry._backends.values() if i["healthy"])
    return {
        "status": "ok",
        "backends_healthy": alive,
        "backends_total": len(registry._backends),
        "models": len(registry._routing),
    }

# ---------------------------------------------------------------------------
# Admin routes (gated by ADMIN_TOKEN, must be before catch-all)
# ---------------------------------------------------------------------------
ADMIN_UI_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>llmconduit aliases</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }
  h1 { font-size: 1.2rem; }
  .role { margin-bottom: 1.5rem; }
  .role h2 { font-size: 1rem; margin: 0 0 0.5rem; }
  .role .target { font-size: 0.8rem; color: #999; margin-bottom: 0.4rem; }
  button { display: block; width: 100%; text-align: left; margin: 2px 0; padding: 6px 10px;
           background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; cursor: pointer; }
  button.active { background: #2a6; border-color: #2a6; color: #fff; }
  #err { color: #f55; }
</style>
</head>
<body>
<h1>llmconduit &mdash; role aliases</h1>
<div id="err"></div>
<div id="roles"></div>
<script>
const token = new URLSearchParams(location.search).get("token") || "";

async function api(path, opts) {
  const resp = await fetch(path, Object.assign({}, opts, {
    headers: Object.assign({"Authorization": "Bearer " + token, "Content-Type": "application/json"}, (opts && opts.headers) || {})
  }));
  if (resp.status === 401) throw new Error("invalid or missing token");
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function render() {
  const errEl = document.getElementById("err");
  const rolesEl = document.getElementById("roles");
  errEl.textContent = "";
  rolesEl.innerHTML = "";
  let data;
  try {
    data = await api("/admin/aliases");
  } catch (e) {
    errEl.textContent = e.message;
    return;
  }
  for (const [name, info] of Object.entries(data.aliases)) {
    const section = document.createElement("div");
    section.className = "role";
    const h2 = document.createElement("h2");
    h2.textContent = name;
    const target = document.createElement("div");
    target.className = "target";
    target.textContent = info.target + (info.resolved ? "" : "  (not currently resolvable)") + (info.source === "override" ? "  [override]" : "");
    section.appendChild(h2);
    section.appendChild(target);
    for (const model of data.available_models) {
      const btn = document.createElement("button");
      btn.textContent = model;
      if (model === info.target) btn.classList.add("active");
      btn.onclick = async () => {
        try {
          await api("/admin/alias/" + encodeURIComponent(name), {
            method: "POST",
            body: JSON.stringify({target: model}),
          });
          render();
        } catch (e) {
          errEl.textContent = e.message;
        }
      };
      section.appendChild(btn);
    }
    rolesEl.appendChild(section);
  }
  if (Object.keys(data.aliases).length === 0) {
    rolesEl.textContent = "No aliases defined in config.yaml yet.";
  }
}

render();
</script>
</body>
</html>
"""

@app.get("/admin/status", dependencies=[Depends(require_admin_auth)])
async def admin_status():
    return registry.status()

@app.post("/admin/rescan", dependencies=[Depends(require_admin_auth)])
async def admin_rescan():
    await registry.full_scan()
    return {"ok": True, "models": len(registry._routing)}

@app.post("/admin/rescan/{backend_name}", dependencies=[Depends(require_admin_auth)])
async def admin_rescan_backend(backend_name: str):
    url = registry.backend_url_for_name(backend_name)
    if not url:
        raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
    await registry.scan_backend(url)
    models = [k for k, v in registry._routing.items() if v == url]
    return {"ok": True, "backend": backend_name, "models": models}

@app.post("/admin/reload", dependencies=[Depends(require_admin_auth)])
async def admin_reload():
    await registry.reload_config()
    return {"ok": True, "backends": len(registry._backends), "models": len(registry._routing)}

@app.get("/admin/aliases", dependencies=[Depends(require_admin_auth)])
async def admin_list_aliases():
    return {
        "aliases": registry.alias_status(),
        "available_models": list(registry._routing.keys()),
    }

@app.post("/admin/alias/{name}", dependencies=[Depends(require_admin_auth)])
async def admin_set_alias(name: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    target = body.get("target")
    if not target:
        raise HTTPException(status_code=400, detail="Missing 'target' field")
    try:
        registry.set_alias_override(name, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "alias": name.upper(), "target": target}

@app.delete("/admin/alias/{name}", dependencies=[Depends(require_admin_auth)])
async def admin_clear_alias(name: str):
    cleared = registry.clear_alias_override(name)
    return {"ok": True, "alias": name.upper(), "cleared": cleared}

@app.get("/admin/ui", response_class=HTMLResponse)
async def admin_ui():
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    return ADMIN_UI_HTML

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    dependencies=[Depends(require_proxy_auth)],
)
async def catch_all(request: Request, path: str):
    """Best-effort fallback: route to first healthy backend."""
    healthy = [u for u, info in registry._backends.items() if info["healthy"]]
    if not healthy:
        raise HTTPException(status_code=503, detail="No healthy backends available")
    return await _proxy(request, healthy[0])
