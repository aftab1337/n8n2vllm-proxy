# n8n to vLLM Translation Proxy

<img alt="banner" src="[https://raw.githubusercontent.com/ludo-technologies/pyscn/main/assets/demo-report.png](https://serintel.s3.us-east-1.amazonaws.com/banner.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAZIF2FA7KRDVY7Q5H%2F20260825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260825T185321Z&X-Amz-Expires=300&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEMaCXVzLWVhc3QtMSJHMEUCIQCahrP59kFRxt1YZEWdUIDc%2FEsYlHmc%2BPJ2cgD3kd3gfgIgeQx5bcihYRRXcNRinDXHnUNm65TmhsNsBW1cw5IpPuEq2gIIDBAAGgw2MzYwNDU1NTk3NjUiDIe7KPUzN6Btm3ETQSq3AlFQEiA1EwiRjEJXgw9C7ubyZAq%2BonJubMqoe5AkMFdv3v05UtBGfYVfSzSdofElCmE4QCK3v3lekNop0Nd0JbiAhI%2FZN6mztjk%2FNn0yLHlJLphJBTqaGQZaI8FBsuH0k9C4st3Ra8N05XkpZqoi38ivuUSYaggxh%2Fqd0ijjoGMxPtBffq112Cs5YmYRRdawEShk1CDBkRiurzghJts7G%2B67jdTllTC4%2FJ2OjW%2BrLwAPbfK6l7DAH%2BfZGGRlNPpzLVSWFq%2FzHw%2FqXH2RrvTbtUiPO96jakul6CHJItXQHJ34gQAGOIqh8NHUsa6s9A9neWdKim3i3oBsn9crnnD35qW9PHpo36UWc9ybkoWwvJziSLafgEGydWmJw%2BZLsIYld%2Bk%2B8q6HZzjYBfYoxzGbUJdIMWODDTRbMPjFt9QGOq0CLSIR3EMl2zD8zjGXdQdj4S5g0kE6bfq93jZTKT7Ois4kRzyIv2wDxIAXvuKVXA%2Fy1Eivk6jJ44x5a%2FJM%2BeYm8XIORW0UgWu1gDoL1LhIdxqwuXvK0WAxe3qu30v1%2Fwnx2UJj26DCJtXGNopx5DO6QTLKaEusDUdIq3vw3Pe6vEt%2B7ruZi31oKP4EAuZZvN4k9xxyXtM8%2Fwo1J7RFtNjA%2Frwp68m%2FerhWEjGlKpi8bgx9uaXyZp%2Br1GGEMgmB%2BMjc250oRMfFyTYBTWd%2FT73h6Mlr%2Fti5emtBMr1M35FRrquyh%2FkaMvPXE3HjZa5dNVDb08gdAH7XI8ZrXPBsdXhlaOC4LTPVquVtW7ae4UYp%2BXZ83Y%2FesundAgySxMXnJR1wVkfHlY2baXRp%2FrKmRw%3D%3D&X-Amz-Signature=1c5292a9e1e13f7d78ff04b2ed6c541a94c7fc7d6e65c70e0d9bc4b75bdce826&X-Amz-SignedHeaders=host&response-content-disposition=inline)" width="720">
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
