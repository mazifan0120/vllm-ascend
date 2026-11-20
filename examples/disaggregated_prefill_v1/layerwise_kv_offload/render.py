#!/usr/bin/env python3
"""Render the supplied launch profiles using only the Python standard library."""

import argparse
import ipaddress
import json
import re
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=["performance", "capacity"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if set(config["nodes"]) != {"p0", "p1", "d0", "d1"} or len(set(config["nodes"].values())) != 4:
        raise ValueError("Provide exactly four distinct node IPs: p0, p1, d0, d1")
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]+", config["nic"]) or config["nic"] == "CHANGE_ME":
        raise ValueError("Set the communication NIC name")
    for node in ["p0", "p1", "d0", "d1"]:
        ipaddress.IPv4Address(config["nodes"][node])
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    (args.output / "cluster.json").write_text(json.dumps(config, indent=2) + "\n")
    for node, ip in config["nodes"].items():
        dest = args.output / node
        dest.mkdir()
        for template in (source / "profiles" / args.profile / node).glob("*.in"):
            text = template.read_text().replace("@NIC@", config["nic"])
            for key, value in config["nodes"].items():
                text = text.replace("@" + key.upper() + "_IP@", value)
            if re.search(r"@[A-Z0-9_]+@", text):
                raise ValueError(f"Unexpanded placeholder: {template}")
            path = dest / template.stem
            path.write_text(text)
            if path.suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True)
        (dest / "node.json").write_text(
            json.dumps(
                {
                    "node": node,
                    "ip": ip,
                    "profile": args.profile,
                    "image": config["image"],
                    "image_id": config["image_id"],
                    "proxy_port": 8499 if args.profile == "performance" else 8497,
                },
                indent=2,
            )
            + "\n"
        )
        shutil.copyfile(source / "run-node.sh", dest / "run-node.sh")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
