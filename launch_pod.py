"""Launch a Runpod H200 pod serving WhiteRabbitNeo V3 via SGLang's OpenAI-compatible API.

    uv run launch_pod.py
"""
import os

import runpod
from dotenv import load_dotenv

load_dotenv()

runpod.api_key = os.environ["RUNPOD_API_KEY"]

MODEL_PATH = "WhiteRabbitNeo/WhiteRabbitNeo-V3-7B"
SERVED_MODEL_NAME = "whiterabbitneo-v3"
PORT = 30000

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

print(f"Pod {pod['id']} created. Weights download on first boot; give it a few minutes.")
print(f"BASE_URL=https://{pod['id']}-{PORT}.proxy.runpod.net/v1")
