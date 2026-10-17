# runpod-self-hosting

Self-host [WhiteRabbitNeo V3](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-V3-7B) behind an
**OpenAI-compatible API** using [SGLang](https://github.com/sgl-project/sglang) on a RunPod **H200** GPU.

WhiteRabbitNeo-V3-7B is a Qwen2.5-Coder-7B–based cybersecurity model with a 32K context window
(Apache-2.0). It weighs ~15 GB in bf16, so a single H200 (141 GB) runs it comfortably at `--tp 1`
with room to spare for a long-context KV cache.

Once running, SGLang exposes OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses
(`/v1/responses`), and Anthropic Messages (`/v1/messages`) on port `30000`.

## Deploy on RunPod with Python (uv)

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # set RUNPOD_API_KEY and API_KEY (optionally HF_TOKEN)
uv run main.py            # uses config.yaml; or: uv run main.py path/to/other.yaml
```

This creates the pod described in `config.yaml` (by default an H200 Secure Cloud pod running
`lmsysorg/sglang:latest` on port `30000`), caches weights on a volume at
`/root/.cache/huggingface`, and streams the pod logs. Once SGLang's `/health` responds it prints
the `BASE_URL` to call, and keeps streaming logs until you press Ctrl+C, which stops the pod.

Running it again reuses the pod with the same `pod.name` (resuming it if stopped) instead of
creating a new one, so the cached weights are kept. If `config.yaml` or `API_KEY` changed since
then, the pod is updated in place (image, SGLang flags, port, disk, API key) and restarted. Only
a change to `gpu_type`, `gpu_count`, `cloud_type` or `volume_gb`, or a pod that can't be resumed
(e.g. no free H200 on its machine), makes it ask whether to delete the old pod and create a new one.

To serve another model or GPU layout, copy `config.yaml` and change the `pod` section (GPU type,
`gpu_count`, disk sizes) and the SGLang flags under `server.args` (e.g. `gpu_count: 8` with
`tp: 8` for an 8×H200 pod).

## Test all agent APIs

With the pod running, test a forced function call through all three protocols:

```bash
uv run src/test_endpoints.py
```

The script discovers the running pod from `config.yaml`. You can instead pass any SGLang URL,
with or without `/v1`:

```bash
uv run src/test_endpoints.py https://<pod-id>-30000.proxy.runpod.net/v1
```

Each test passes only if the response contains a structured `get_weather({"city":"Paris"})` tool
call—not merely an HTTP 200 or plain-text imitation.

## Deploy on RunPod (console, no custom image)

1. Create a **GPU Pod** and pick an **H200 SXM** instance.
2. Set the **Container Image** to `lmsysorg/sglang:latest`.
3. Under **Expose HTTP Ports**, add `30000`.
4. Set the **Container Start Command** to:

   ```
   python3 -m sglang.launch_server --model-path WhiteRabbitNeo/WhiteRabbitNeo-V3-7B --served-model-name whiterabbitneo-v3 --host 0.0.0.0 --port 30000 --context-length 32768 --mem-fraction-static 0.9 --tp 1 --api-key YOUR_API_KEY
   ```

5. (Optional) Add a `HF_TOKEN` environment variable for faster/gated downloads.
6. Deploy. The first boot downloads the weights (~15 GB); wait for the log line
   `The server is fired up and ready to roll!`.

RunPod gives the HTTP port a public proxy URL of the form
`https://<pod-id>-30000.proxy.runpod.net`. That is your API base host.

## Deploy on RunPod (custom image)

Build and push the image in this repo (it bakes in `start.sh`, which reads the env vars below),
then point the Pod's **Container Image** at it and leave the start command empty:

```bash
docker build -t <your-registry>/whiterabbitneo-sglang:latest .
docker push <your-registry>/whiterabbitneo-sglang:latest
```

## Run locally (own H200 box)

Requires Docker with the NVIDIA Container Toolkit.

```bash
cp .env.example .env   # set API_KEY (optionally HF_TOKEN)
docker compose up
```

## Call the API

```bash
curl https://<pod-id>-30000.proxy.runpod.net/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "whiterabbitneo-v3",
    "messages": [{"role": "user", "content": "Explain a SYN flood in two sentences."}]
  }'
```

With the OpenAI SDK:

```bash
BASE_URL=https://<pod-id>-30000.proxy.runpod.net/v1 uv run src/test_api.py
```

`test_api.py` reads `API_KEY` from `.env`; other clients pass it as the OpenAI `api_key` or an
`Authorization: Bearer <key>` header.

## Configuration

`start.sh` and `docker-compose.yml` read these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `WhiteRabbitNeo/WhiteRabbitNeo-V3-7B` | Hugging Face model to serve |
| `SERVED_MODEL_NAME` | `whiterabbitneo-v3` | Name clients pass as `model` |
| `PORT` | `30000` | HTTP port |
| `CONTEXT_LENGTH` | `32768` | Max context window |
| `MEM_FRACTION_STATIC` | `0.9` | Fraction of VRAM reserved for weights + KV cache |
| `TP_SIZE` | `1` | Tensor-parallel GPUs (1 is enough on an H200) |
| `API_KEY` | _(required)_ | Clients must send `Authorization: Bearer <key>` |
| `HF_TOKEN` | _(unset)_ | Hugging Face token for gated/faster downloads |
