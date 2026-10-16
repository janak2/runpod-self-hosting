"""Create a Runpod H200 pod serving WhiteRabbitNeo V3 via SGLang, stream its logs, and wait for it."""
import json
import os
import time

import requests
import runpod

MODEL_PATH = "WhiteRabbitNeo/WhiteRabbitNeo-V3-7B"
SERVED_MODEL_NAME = "whiterabbitneo-v3"
PORT = 30000


def create_pod() -> str:
    """Create the pod and return its ID."""
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    args = [
        "python3 -m sglang.launch_server",
        f"--model-path {MODEL_PATH}",
        f"--served-model-name {SERVED_MODEL_NAME}",
        "--host 0.0.0.0",
        f"--port {PORT}",
        "--context-length 32768",
        "--mem-fraction-static 0.9",
        "--tp 1",
    ]
    if os.environ.get("API_KEY"):
        args.append(f"--api-key {os.environ['API_KEY']}")

    env = {"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else None

    pod = runpod.create_pod(
        name="whiterabbitneo-sglang",
        image_name="lmsysorg/sglang:latest",
        gpu_type_id="NVIDIA H200",
        gpu_count=1,
        cloud_type="SECURE",
        ports=f"{PORT}/http",
        container_disk_in_gb=60,
        volume_in_gb=60,
        volume_mount_path="/root/.cache/huggingface",
        docker_args=" ".join(args),
        env=env,
    )
    return pod["id"]


def base_url(pod_id: str) -> str:
    return f"https://{pod_id}-{PORT}.proxy.runpod.net"


def stream_logs(pod_id: str) -> None:
    """Print the pod's container and system logs forever, resuming where it left off if the stream drops."""
    headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}
    while True:
        try:
            with requests.get(
                f"https://api.runpod.io/v2/pods/{pod_id}/logs",
                headers=headers,
                stream=True,
                timeout=(10, 300),
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    line = raw.decode("utf-8")
                    if line.startswith("id:"):
                        headers["Last-Event-ID"] = line[3:].strip()
                    elif line.startswith("data:"):
                        event = json.loads(line[5:])
                        print(f"[{event['source']}] {event['line']}", flush=True)
        except requests.RequestException:
            pass
        time.sleep(5)


def wait_until_ready(pod_id: str) -> None:
    """Block until SGLang's /health returns 200 through the Runpod proxy."""
    api_key = os.environ.get("API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while True:
        try:
            if requests.get(f"{base_url(pod_id)}/health", headers=headers, timeout=10).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(10)
