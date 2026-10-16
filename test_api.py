"""Smoke-test the OpenAI-compatible endpoint.

    uv run --env-file .env test_api.py

Set BASE_URL to your RunPod proxy URL, e.g.
    BASE_URL=https://<pod-id>-30000.proxy.runpod.net/v1
"""
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BASE_URL", "http://localhost:30000/v1"),
    api_key=os.environ.get("API_KEY") or "EMPTY",
)

resp = client.chat.completions.create(
    model=os.environ.get("MODEL", "whiterabbitneo-v3"),
    messages=[
        {"role": "user", "content": "Write a Python function that scans a host for open TCP ports."},
    ],
    temperature=0.7,
    max_tokens=512,
)

print(resp.choices[0].message.content)
