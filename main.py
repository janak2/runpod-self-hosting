"""Launch WhiteRabbitNeo V3 on a Runpod H200, stream the pod logs, and wait until the API is ready.

    uv run main.py
"""
import threading

from dotenv import load_dotenv

from src.pod import base_url, create_pod, stream_logs, wait_until_ready

load_dotenv()

pod_id = create_pod()
print(f"Pod {pod_id} created. Streaming logs until the server is ready...", flush=True)

threading.Thread(target=stream_logs, args=(pod_id,), daemon=True).start()
wait_until_ready(pod_id)

print(f"\nServer is ready.\nBASE_URL={base_url(pod_id)}/v1")
