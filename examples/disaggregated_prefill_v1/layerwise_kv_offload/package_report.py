#!/usr/bin/env python3
"""Build the self-contained report bundle from version-controlled inputs."""

import argparse
import hashlib
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build/reports/layerwise-kv-offload-report-20261120.zip")
    args = parser.parse_args()
    assets = root / "docs/source/assets/layerwise_kv_offload"
    text = (root / "docs/source/user_guide/feature_guide/layerwise_kv_offload.md").read_text()
    text = text.replace(
        "../../assets/layerwise_kv_offload/performance-overview.png", "figures/performance-overview.png"
    )
    files = {"report.md": text.encode()}
    for path in assets.iterdir():
        if path.is_file():
            files["figures/" + path.name] = path.read_bytes()
    for source, target in [
        (root / "examples/disaggregated_prefill_v1/layerwise_kv_offload", "runtime"),
        (assets / "evidence", "results"),
    ]:
        for path in source.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[target + "/" + path.relative_to(source).as_posix()] = path.read_bytes()
    files["README.txt"] = (
        b"Read report.md first. Mermaid renders the design diagrams; PNG/SVG performance figures are included.\n"
        b"runtime/ contains standalone Python/Bash/Docker configuration and test tools; read runtime/README.md.\n"
        b"The feature image archive and model weights must be supplied separately. "
        b"The local image alias is not a public download URL.\n"
        b"results/ contains compact original measurement evidence. "
        b"Portable wrappers and the reference HTTP client have local checks only, as documented.\n"
    )
    files["MANIFEST.sha256"] = "".join(
        hashlib.sha256(data).hexdigest() + "  " + name + "\n" for name, data in sorted(files.items())
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Report archive integrity check failed")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    Path(str(args.output) + ".sha256").write_text(digest + "  " + args.output.name + "\n")
    print(f"{args.output}: {len(files)} files, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
