#!/usr/bin/env python3
"""Check the deployed layerwise APIs, deterministic requests and TP receipts."""

import argparse
import json
import re
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Lock

HOST_TARGET_EVENT = re.compile(r"Layerwise Host target (ready|closed) request=(\S+) tp=(\d+)")


def audit_receipts(args, results):
    deadline = time.monotonic() + 120
    while True:
        lines = []
        for log in args.logs.glob("d*/logs/vllm_decode_dp*.log"):
            lines.extend((str(log), line) for line in log.read_text(errors="replace").splitlines())
        covered = set()
        pending = []
        for result in results:
            if args.layerwise:
                events = [
                    (path, match[1], int(match[3]))
                    for path, line in lines
                    if (match := HOST_TARGET_EVENT.search(line)) and result["id"] in match[2]
                ]
                engines = {path for path, state, _ in events if state == "ready"}
                ready_counts = {count for _, state, count in events if state == "ready"}
                ranks = set(range(next(iter(ready_counts)))) if len(ready_counts) == 1 else set()
                retired = any(
                    path in engines and state == "closed" and count == args.tp for path, state, count in events
                )
            else:
                receipts = [
                    (path, line)
                    for path, line in lines
                    if result["id"] in line and "published after prepare_decode" in line
                ]
                ranks = {int(match[1]) for _, line in receipts if (match := re.search(r"_TP(\d+)_EP", line))}
                engines = {path for path, _ in receipts}
                retired = any(
                    path in engines and result["id"] in line and "Released target for" in line for path, line in lines
                )
            if len(engines) != 1 or ranks != set(range(args.tp)) or not retired:
                pending.append(
                    {
                        "id": result["id"],
                        "engines": sorted(engines),
                        "ranks": sorted(ranks),
                        "retired": retired,
                    }
                )
            covered.update(engines)
        if not pending:
            assert len(covered) == args.expected_decode_engines, covered
            print(
                json.dumps(
                    {
                        "passed": len(results),
                        "decode_engines": sorted(covered),
                        "tp_receipts": args.tp,
                    }
                ),
                flush=True,
            )
            return
        if time.monotonic() >= deadline:
            raise AssertionError(pending)
        time.sleep(5)


def audit_plain_transfers(args, results):
    ranks_by_request = {result["id"]: set() for result in results}
    pattern = re.compile(r"_TP(\d+)_EP.*Sending transmitting signal b'(done|failed)_sending_msg' for request (\S+)")
    for log in args.logs.glob("p*/logs/vllm_prefill_dp*.log"):
        with log.open(errors="replace") as lines:
            for line in lines:
                match = pattern.search(line)
                if match is None:
                    continue
                rank, signal, request = match.groups()
                if request in ranks_by_request:
                    assert signal == "done", line
                    ranks_by_request[request].add(int(rank))
    expected = set(range(args.tp))
    missing = {request: sorted(ranks) for request, ranks in ranks_by_request.items() if ranks != expected}
    assert not missing, missing
    print(json.dumps({"passed": len(results), "plain_done_sending_tp": args.tp}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://172.168.4.237:8333")
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--tp", default=8, type=int)
    parser.add_argument("--expected-decode-engines", default=4, type=int)
    parser.add_argument("--audit-results", type=Path)
    parser.add_argument("--run-nonce")
    parser.add_argument("--cache-salt")
    parser.add_argument("--plain-layerwise", action="store_true")
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--skip-log-audit", action="store_true")
    args = parser.parse_args()
    audit = audit_plain_transfers if args.plain_layerwise else audit_receipts
    if args.audit_results:
        records = [json.loads(line) for line in args.audit_results.read_text().splitlines()]
        # Support both smoke JSONL (including its summary) and long probe phases.
        results = [r for r in records if r.get("id") and r.get("phase") in (None, "complete")]
        if not results:
            raise ValueError("Audit input contains no completed request records")
        audit(args, results)
        return
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    run_nonce = args.run_nonce or uuid.uuid4().hex
    output_lock = Lock()

    def request(case):
        label, repetitions = case
        replay = label.startswith("replay-")
        nonce = run_nonce + label.removeprefix("replay-")
        expected = "QUARTZ" if replay else "MAPLE"
        prompt = (
            f"Document {nonce}.\nThe first reference word is MAPLE.\n"
            + "Mooncake stores KV blocks as objects.\n" * repetitions
            + "\nThe second reference word is QUARTZ.\n"
            + ("Reply only with the second reference word." if replay else "Reply only with the first reference word.")
        )
        body = {
            "model": "glm-52",
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        }
        if args.cache_salt:
            body["cache_salt"] = args.cache_salt
        start = time.monotonic()
        req = urllib.request.Request(
            args.proxy + "/v1/chat/completions",
            json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=900) as response:
            data = json.load(response)
        choice = data["choices"][0]
        answer = choice["message"].get("content", "").strip()
        result = {
            "case": label,
            "id": data["id"],
            "run_nonce": run_nonce,
            "input_sha256": sha256(prompt.encode()).hexdigest(),
            "answer": answer,
            "usage": data["usage"],
            "finish_reason": choice["finish_reason"],
            "seconds": round(time.monotonic() - start, 3),
        }
        with output_lock:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        correct = bool(re.fullmatch(rf"{expected}[.!。！]?", answer)) and choice["finish_reason"] == "stop"
        if repetitions > 2000:
            correct = correct and data["usage"]["prompt_tokens"] > 16384
        return result, correct

    # Exercise every D engine, including deployments with different P/D DP.
    decode_engines = args.expected_decode_engines
    results = [request((f"short-{rank}", 60)) for rank in range(decode_engines)]
    multichunk = [(f"multichunk-{rank}", 4200) for rank in range(decode_engines)]
    with ThreadPoolExecutor(max_workers=decode_engines) as pool:
        results.extend(pool.map(request, multichunk))
    results.extend(request((f"replay-{label}", repetitions)) for label, repetitions in multichunk)
    if not args.skip_log_audit:
        audit(args, [result for result, _ in results])
    mismatches = [result for result, correct in results if not correct]
    if mismatches:
        raise AssertionError(mismatches)


if __name__ == "__main__":
    main()
