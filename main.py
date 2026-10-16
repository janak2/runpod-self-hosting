"""Launch the pod described by a config file, stream its logs, and wait until the API is ready.

    uv run main.py [config.yaml]
"""
import argparse
import os
import sys
import threading
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.pod import base_url, create_pod, stream_logs, wait_until_ready

load_dotenv()
if not os.environ.get("API_KEY"):
    sys.exit("API_KEY must be set in .env")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", nargs="?", default="config.yaml", help="pod and SGLang server config")
config = yaml.safe_load(Path(parser.parse_args().config).read_text())
port = config["server"]["port"]

pod_id = create_pod(config)
print(f"Pod {pod_id} created. Streaming logs until the server is ready...", flush=True)

threading.Thread(target=stream_logs, args=(pod_id,), daemon=True).start()
wait_until_ready(pod_id, port)

print(f"\nServer is ready.\nBASE_URL={base_url(pod_id, port)}/v1")
