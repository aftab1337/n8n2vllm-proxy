# n8n to vLLM Translation Proxy

A high-performance, asynchronous translation proxy built with FastAPI and `httpx`. It bridges n8n's built-in Instance AI sidebar (powered by Vercel AI SDK) with standard OpenAI-compatible LLM servers like vLLM, Ollama, Aphrodite, or LocalAI.

---

## The Problem

n8n's Instance AI Assistant uses the Vercel AI SDK, which often routes requests to OpenAI's proprietary Responses API endpoint (`/v1/responses`) using a custom `input` array schema instead of the standard Chat Completions schema (`/v1/chat/completions` with `messages`).

When connecting n8n to self-hosted LLM backends (such as vLLM running on Modal, RunPod, or local GPUs), requests fail with `500 Internal Server Error`, `404 Not Found`, or schema validation errors.

---

## The Solution

This proxy acts as a transparent middleware layer:

1. **Dual Endpoint Support**: Handles both `/v1/responses` and standard `/v1/chat/completions`.
2. **Schema Translation**: Translates `instructions` and `input` objects into standard `messages` arrays, normalizing tool definitions and parameters.
3. **Bi-Directional SSE Streaming**: Converts upstream vLLM completion chunks into typed Server-Sent Events (`response.created`, `response.output_text.delta`, `response.completed`) required by the Vercel AI SDK.
4. **Model Name Resolution**: Cleans provider prefixes (like `openai/`) and enforces your loaded model identifier.

---

## Architecture

```text
┌─────────────────────────┐
│    n8n Instance AI      │
│  (Vercel AI SDK Client) │
└───────────┬─────────────┘
            │  POST /v1/responses or /v1/chat/completions
            ▼
┌─────────────────────────┐
│   FastAPI Proxy Layer   │ <── Transforms Schema, Handles SSE Protocol
└───────────┬─────────────┘
            │  POST /v1/chat/completions
            ▼
┌─────────────────────────┐
│   vLLM Engine Server    │
│  (e.g., Qwen 2.5 32B)   │
└─────────────────────────┘
```

---

## Project Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── proxy.py
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/n8n-vllm-proxy.git
cd n8n-vllm-proxy
```

### 2. Configure docker-compose.yml

Edit `docker-compose.yml` to specify your upstream vLLM backend and model:

```yaml
services:
  vllm-proxy:
    build: .
    container_name: n8n-vllm-proxy
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - VLLM_BACKEND_URL=https://your-modal-app.modal.run
      - DEFAULT_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
      - OVERRIDE_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
      # - VLLM_API_KEY=your-backend-key  # Optional
      - LOG_LEVEL=INFO
    networks:
      - n8n-network

networks:
  n8n-network:
    name: n8n-network
```

### 3. Build and Start the Proxy

```bash
docker compose up -d --build
```

### 4. Verify Proxy Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "n8n-vllm-proxy",
  "target_vllm_endpoint": "https://your-modal-app.modal.run/v1",
  "target_model": "Qwen/Qwen2.5-32B-Instruct-AWQ"
}
```

---

## Configuring n8n

Add the following environment variables to your n8n configuration (inside your n8n `docker-compose.yml` or `.env` file):

```env
# Enable the AI Assistant module
N8N_ENABLED_MODULES=instance-ai

# Set model identifier
N8N_INSTANCE_AI_MODEL=openai/Qwen/Qwen2.5-32B-Instruct-AWQ
N8N_INSTANCE_AI_MODEL_API_KEY=not-needed

# Point to your proxy endpoint (Notice the /v1 suffix)
N8N_INSTANCE_AI_MODEL_URL=https://vllm-proxy.yourdomain.com/v1
OPENAI_BASE_URL=https://vllm-proxy.yourdomain.com/v1

# Required when running behind an Nginx reverse proxy
N8N_PROXY_HOPS=1
```

Restart your n8n instance:
```bash
docker compose up -d
```

---

## Nginx Reverse Proxy Setup (Recommended for Custom Domains)

When exposing the proxy behind Nginx with SSL, ensure response buffering is disabled so Server-Sent Events stream without delay:

```nginx
server {
    listen 80;
    server_name vllm-proxy.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Critical for real-time LLM token streaming
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        # Extended timeouts for large model inference
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Enable SSL via Certbot:
```bash
certbot --nginx -d vllm-proxy.yourdomain.com
```

---

## Configuration Options

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_BACKEND_URL` | `http://localhost:8000` | Base URL of your upstream vLLM server. |
| `VLLM_API_KEY` | *(Empty)* | API Key sent in the `Authorization` header to upstream vLLM. |
| `DEFAULT_MODEL` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | Fallback model name if none is provided. |
| `OVERRIDE_MODEL` | *(Empty)* | If set, forces all incoming requests to use this model identifier. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PORT` | `8000` | Port for the proxy server. |
| `HOST` | `0.0.0.0` | Host interface binding. |

---

## Troubleshooting

### 1. 404 Not Found on `/v1/chat/completions`
Ensure you rebuilt your Docker container using `docker compose up -d --build` so the latest route handlers are compiled into the container image.

### 2. Stream Stalls or Delays
If tokens appear in batches rather than streaming smoothly, check your reverse proxy (Nginx, Cloudflare, Caddy). Verify that `proxy_buffering off;` and `X-Accel-Buffering: no` headers are present.

### 3. Rate Limit or Express Proxy Warnings in n8n
If n8n logs show `ValidationError: The 'X-Forwarded-For' header is set`, add `N8N_PROXY_HOPS=1` to n8n's environment variables.

---

## License

MIT License. Free to use and modify for personal and commercial deployments.
