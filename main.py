"""Launch the pod described by a config file and stream its logs until Ctrl+C, which stops the pod.

    uv run main.py [config.yaml]
"""
import argparse
import os
import sys
import threading
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.pod import base_url, create_pod, stop_pod, stream_logs, wait_until_ready

load_dotenv()
if not os.environ.get("API_KEY"):
    sys.exit("API_KEY must be set in .env")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", nargs="?", default="config.yaml", help="pod and SGLang server config")
config = yaml.safe_load(Path(parser.parse_args().config).read_text())
port = config["server"]["port"]

pod_id = create_pod(config)
print(f"Pod {pod_id} created. Streaming logs; press Ctrl+C to stop the pod.", flush=True)


def announce_ready() -> None:
    wait_until_ready(pod_id, port)
    print(f"\nServer is ready.\nBASE_URL={base_url(pod_id, port)}/v1\n", flush=True)


threading.Thread(target=announce_ready, daemon=True).start()
try:
    stream_logs(pod_id)
except KeyboardInterrupt:
    print(f"\nStopping pod {pod_id}...", flush=True)
    stop_pod(pod_id)
    print("Pod stopped.")
