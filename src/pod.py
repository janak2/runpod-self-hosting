"""Create a Runpod pod serving a model via SGLang, stream its logs, and wait for it."""
import hashlib
import json
import os
import time

import requests
import runpod
from runpod.error import QueryError

HF_CACHE_PATH = "/root/.cache/huggingface"


def server_command(server: dict) -> str:
    """Build the SGLang launch command from the config's `server` section."""
    parts = ["python3 -m sglang.launch_server", "--host 0.0.0.0", f"--port {server['port']}"]
    for flag, value in server["args"].items():
        if value is True:
            parts.append(f"--{flag}")
        elif value is not False:
            parts.append(f"--{flag} {value}")
    parts.append(f"--api-key {os.environ['API_KEY']}")
    return " ".join(parts)


def config_hash(config: dict) -> str:
    """Fingerprint of the config and API_KEY, stored on the pod so a later run can tell if either changed."""
    data = json.dumps(config, sort_keys=True) + os.environ["API_KEY"]
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def confirm(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def replace_pod(pod_id: str, config: dict) -> str:
    """Delete `pod_id` and create a fresh pod from `config`."""
    runpod.terminate_pod(pod_id)
    print(f"Deleted pod {pod_id}.", flush=True)
    return create_pod(config)


def start_pod(config: dict) -> str:
    """Attach to, resume, or create the pod named in `config`, asking before replacing an outdated or unresumable one."""
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    name = config["pod"]["name"]
    existing = next((p for p in runpod.get_pods() if p["name"] == name), None)
    if existing is None:
        return create_pod(config)

    pod_id = existing["id"]
    if f"CONFIG_HASH={config_hash(config)}" not in (existing["env"] or []):
        if confirm(
            f"Pod {pod_id} ({name}) was created with a different config.yaml or API_KEY. "
            "Delete it (and its cached weights) and create a new one?"
        ):
            return replace_pod(pod_id, config)
        print("Keeping the old pod with its old settings.", flush=True)

    if existing["desiredStatus"] == "RUNNING":
        print(f"Pod {pod_id} ({name}) is already running; attaching.", flush=True)
        return pod_id

    try:
        runpod.resume_pod(pod_id, existing["gpuCount"])
    except QueryError as e:
        print(f"Could not resume pod {pod_id}: {e}", flush=True)
        if not confirm("Delete it and create a new one?"):
            raise SystemExit(1)
        return replace_pod(pod_id, config)
    print(f"Resumed existing pod {pod_id} ({name}).", flush=True)
    return pod_id


def create_pod(config: dict) -> str:
    """Create the pod described by `config` and return its ID."""
    pod_config = config["pod"]
    server = config["server"]

    env = {"CONFIG_HASH": config_hash(config)}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    pod = runpod.create_pod(
        name=pod_config["name"],
        image_name=pod_config["image"],
        gpu_type_id=pod_config["gpu_type"],
        gpu_count=pod_config["gpu_count"],
        cloud_type=pod_config["cloud_type"],
        ports=f"{server['port']}/http",
        container_disk_in_gb=pod_config["container_disk_gb"],
        volume_in_gb=pod_config["volume_gb"],
        volume_mount_path=HF_CACHE_PATH,
        docker_args=server_command(server),
        env=env,
    )
    print(f"Created pod {pod['id']}.", flush=True)
    return pod["id"]


def stop_pod(pod_id: str) -> None:
    runpod.stop_pod(pod_id)


def base_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


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


def wait_until_ready(pod_id: str, port: int) -> None:
    """Block until SGLang's /health returns 200 through the Runpod proxy."""
    headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}
    while True:
        try:
            if requests.get(f"{base_url(pod_id, port)}/health", headers=headers, timeout=10).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(10)
