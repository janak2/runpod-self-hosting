"""Create a Runpod pod serving a model via SGLang, stream its logs, and wait for it."""
import hashlib
import json
import os
import time

import requests
import runpod
from runpod.error import QueryError

HF_CACHE_PATH = "/root/.cache/huggingface"
API_V2 = "https://api.runpod.io/v2"
# Pod settings Runpod can't change on an existing pod; changing any of them needs a new pod.
FIXED_POD_KEYS = ("gpu_type", "gpu_count", "cloud_type", "volume_gb")


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


def fingerprint(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def pod_env(config: dict) -> dict:
    """Pod env vars, including fingerprints that let a later run tell what changed since the pod was set up."""
    env = {
        "CONFIG_HASH": fingerprint([config, os.environ["API_KEY"]]),
        "HARDWARE_HASH": fingerprint({key: config["pod"][key] for key in FIXED_POD_KEYS}),
    }
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return env


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}


def update_pod(pod_id: str, config: dict) -> None:
    """Apply the config's changeable settings (image, start command, ports, disk, env) to an existing pod."""
    resp = requests.patch(
        f"{API_V2}/pods/{pod_id}",
        headers=auth_headers(),
        json={
            "image": config["pod"]["image"],
            "args": server_command(config["server"]),
            "ports": [f"{config['server']['port']}/http"],
            "disk": config["pod"]["container_disk_gb"],
            "env": pod_env(config),
        },
        timeout=60,
    )
    if not resp.ok:
        raise SystemExit(f"Updating pod {pod_id} failed: {resp.status_code} {resp.text}")


def restart_pod(pod_id: str) -> None:
    resp = requests.post(f"{API_V2}/pods/{pod_id}/action", headers=auth_headers(), json={"action": "restart"}, timeout=60)
    # 409 means the pod is no longer RUNNING, i.e. the update already restarted it.
    if not resp.ok and resp.status_code != 409:
        raise SystemExit(f"Restarting pod {pod_id} failed: {resp.status_code} {resp.text}")


def confirm(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def replace_pod(pod_id: str, config: dict) -> str:
    """Delete `pod_id` and create a fresh pod from `config`."""
    runpod.terminate_pod(pod_id)
    print(f"Deleted pod {pod_id}.", flush=True)
    return create_pod(config)


def start_pod(config: dict) -> str:
    """Attach to, update, resume, or create the pod named in `config`, asking before replacing one."""
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    name = config["pod"]["name"]
    existing = next((p for p in runpod.get_pods() if p["name"] == name), None)
    if existing is None:
        return create_pod(config)

    pod_id = existing["id"]
    current_env = existing["env"] or []
    wanted_env = pod_env(config)
    running = existing["desiredStatus"] == "RUNNING"

    if f"HARDWARE_HASH={wanted_env['HARDWARE_HASH']}" not in current_env:
        if confirm(
            f"Pod {pod_id} ({name}) has a different GPU type, GPU count, cloud type or volume size, "
            "which can't be changed in place. Delete it (and its cached weights) and create a new one?"
        ):
            return replace_pod(pod_id, config)
        print("Keeping the old pod with its old settings.", flush=True)
    elif f"CONFIG_HASH={wanted_env['CONFIG_HASH']}" not in current_env:
        update_pod(pod_id, config)
        print(f"config.yaml or API_KEY changed; updated pod {pod_id} in place.", flush=True)
        if running:
            restart_pod(pod_id)
            print(f"Restarted pod {pod_id} to apply the changes.", flush=True)
            return pod_id

    if running:
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
        env=pod_env(config),
    )
    print(f"Created pod {pod['id']}.", flush=True)
    return pod["id"]


def stop_pod(pod_id: str) -> None:
    runpod.stop_pod(pod_id)


def base_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def stream_logs(pod_id: str) -> None:
    """Print the pod's container and system logs forever, resuming where it left off if the stream drops.

    Consecutive identical lines (e.g. Docker's per-tick "Extracting" events) are collapsed into one
    line whose repeat count is rewritten in place.
    """
    headers = auth_headers()
    last, count = None, 0
    while True:
        try:
            with requests.get(
                f"{API_V2}/pods/{pod_id}/logs",
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
                        text = f"[{event['source']}] {event['line']}"
                        # The current line is left unterminated so a repeat can redraw it with \r.
                        if text == last:
                            count += 1
                            print(f"\r{text} (x{count})", end="", flush=True)
                        else:
                            print(text if last is None else f"\n{text}", end="", flush=True)
                            last, count = text, 1
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
