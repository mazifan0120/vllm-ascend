#!/usr/bin/env python3
"""Verify a real near-1M prompt through the layerwise P/D proxy."""

import argparse
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:8497")
    parser.add_argument("--tokenize", default="http://127.0.0.1:8410")
    parser.add_argument("--repetitions", type=int, default=130900)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        # Never silently overwrite the evidence from an earlier attempt.
        args.output.touch(exist_ok=False)

    def emit(data):
        line = json.dumps(data)
        print(line, flush=True)
        if args.output:
            with args.output.open("a") as output:
                output.write(line + "\n")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    prompt = (
        f"Document {uuid.uuid4().hex}.\nThe first reference word is MAPLE.\n"
        + "Mooncake stores KV blocks as objects.\n" * args.repetitions
        + "\nThe second reference word is QUARTZ.\n"
        + "Reply only with the first reference word."
    )
    body = {
        "model": "glm-52",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def post(url, data, timeout):
        request = urllib.request.Request(
            url,
            json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        return opener.open(request, timeout=timeout)

    with post(args.tokenize + "/tokenize", dict(body, add_generation_prompt=True), 120) as response:
        count = json.load(response)["count"]
    assert 1040000 <= count <= 1048544, count
    emit({"phase": "tokenized", "prompt_tokens": count})
    body.update(
        temperature=0,
        max_tokens=32,
        stream=True,
        stream_options={"include_usage": True},
    )
    start = time.monotonic()
    answer = []
    usage = first = request_id = finish = None
    with post(args.proxy + "/v1/chat/completions", body, args.timeout) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            assert "error" not in event, event
            if request_id is None and event.get("id"):
                request_id = event["id"]
                emit({"phase": "stream_open", "id": request_id})
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first is None:
                        first = time.monotonic() - start
                        emit({"phase": "first_token", "id": request_id, "seconds": first})
                    answer.append(content)
                finish = choice.get("finish_reason") or finish
    result = {
        "phase": "complete",
        "id": request_id,
        "prompt_tokens": count,
        "answer": "".join(answer),
        "usage": usage,
        "finish_reason": finish,
        "ttft_seconds": first,
        "seconds": time.monotonic() - start,
    }
    emit(result)
    assert usage is not None and usage["prompt_tokens"] == count, result
    assert re.fullmatch(r"MAPLE[.!。！]?", result["answer"].strip()), result
    assert finish == "stop", result


if __name__ == "__main__":
    main()
