#!/usr/bin/env python3
"""Dependency-free SSE load client for an existing prepared JSONL workload."""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=55000,
        help="Set 0 to skip the exact input count assertion for prefix warmup",
    )
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError("Positive concurrency and max-tokens are required")
    cases = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("No input requests")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(case):
        start = time.perf_counter()
        record = {"group": case.get("group"), "success": False}
        body = {
            "model": "glm-52",
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if "cache_salt" in case:
            body["cache_salt"] = case["cache_salt"]
        try:
            req = urllib.request.Request(
                args.url.rstrip("/") + "/v1/chat/completions",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            first, usage, done, finish = None, None, False, None
            with opener.open(req, timeout=args.timeout) as response:
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        continue
                    event = json.loads(data)
                    if event.get("error") or event.get("object") == "error":
                        raise RuntimeError(event)
                    record["id"] = event.get("id", record.get("id"))
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        if first is None and choice.get("delta", {}).get("content"):
                            first = time.perf_counter() - start
                        finish = choice.get("finish_reason") or finish
            elapsed = time.perf_counter() - start
            if not (done and first is not None and usage and finish in ("length", "stop")):
                raise RuntimeError("Missing content, usage or complete SSE termination")
            if usage["completion_tokens"] != args.max_tokens:
                raise RuntimeError(f"Output token count mismatch: {usage}")
            if args.input_tokens and usage["prompt_tokens"] != args.input_tokens:
                raise RuntimeError(f"Input token count mismatch: {usage}")
            record.update(
                success=True,
                ttft_s=first,
                e2e_s=elapsed,
                usage=usage,
                finish=finish,
                tpot_ms=(elapsed - first) * 1000 / (args.max_tokens - 1) if args.max_tokens > 1 else None,
            )
        except Exception as error:
            record["error"] = str(error)
        return record

    rows = []
    with args.output.open("x") as output, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        start = time.perf_counter()
        futures = [pool.submit(request, case) for case in cases]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
        duration = time.perf_counter() - start
    passed = [r for r in rows if r["success"]]
    summary = {"completed": len(rows), "passed": len(passed), "duration_s": duration}
    if passed:
        ttfts = sorted(r["ttft_s"] for r in passed)

        def percentile(q):
            position = (len(ttfts) - 1) * q
            low = int(position)
            return ttfts[low] + (ttfts[min(low + 1, len(ttfts) - 1)] - ttfts[low]) * (position - low)

        summary.update(
            ttft_mean_s=statistics.mean(ttfts),
            ttft_p50_s=percentile(0.5),
            ttft_p99_s=percentile(0.99),
            output_tokens_s=sum(r["usage"]["completion_tokens"] for r in passed) / duration,
        )
        if args.max_tokens > 1:
            summary["tpot_mean_ms"] = statistics.mean(r["tpot_ms"] for r in passed)
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if len(passed) != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
