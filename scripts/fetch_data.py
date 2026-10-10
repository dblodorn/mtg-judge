#!/usr/bin/env python3
"""Restore a published index + corpus snapshot (see publish_data.py) and verify it.

  python scripts/fetch_data.py                                   # latest, from the default folder
  python scripts/fetch_data.py --source ~/Dropbox/mtg-judge-artifacts --version 2026-10-09_cr...
  python scripts/fetch_data.py --source https://<host>/mtg-judge-artifacts   # any HTTP(S) base URL

Every file is checked against the manifest's sha256 before anything is unpacked,
and the current index/ is replaced only after a successful download. Use this on
the Pi instead of building the index there (embedding takes hours on a Pi).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

sys.path.insert(0, ".")
from src.config import load_config

DEFAULT_ROOT = "~/Dropbox/mtg-judge-artifacts"


class Source:
    """Read files from a local folder or an HTTP(S) base URL."""

    def __init__(self, root: str):
        self.url = root if root.startswith(("http://", "https://")) else None
        self.path = None if self.url else Path(os.path.expanduser(root))

    def read_json(self, rel: str) -> dict:
        if self.url:
            resp = requests.get(f"{self.url.rstrip('/')}/{rel}", timeout=60)
            resp.raise_for_status()
            return resp.json()
        return json.loads((self.path / rel).read_text())

    def download(self, rel: str, dest: Path) -> None:
        if self.url:
            with requests.get(f"{self.url.rstrip('/')}/{rel}", stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for block in resp.iter_content(1 << 20):
                        f.write(block)
        else:
            shutil.copy2(self.path / rel, dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract(archive: Path, dest_root: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):  # Python ≥ 3.11.4: blocks path traversal, links out
            tar.extractall(dest_root, filter="data")
            return
        # Raspberry Pi OS Bookworm ships 3.11.2 — check member paths ourselves
        root = dest_root.resolve()
        for member in tar.getmembers():
            target = (dest_root / member.name).resolve()
            if not (target == root or root in target.parents) or member.issym() or member.islnk():
                sys.exit(f"✗ Unsafe path in archive: {member.name}")
        tar.extractall(dest_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config")
    parser.add_argument("--source", default=os.environ.get("MTG_JUDGE_ARTIFACTS", DEFAULT_ROOT))
    parser.add_argument("--version", help="default: latest.json")
    parser.add_argument("--index-only", action="store_true", help="skip the corpus snapshot")
    args = parser.parse_args()
    config = load_config(args.config)

    source = Source(args.source)
    version = args.version or source.read_json("latest.json")["version"]
    manifest = source.read_json(f"{version}/manifest.json")
    if manifest["embed_model"] != config["embed_model"]:
        sys.exit(f"✗ Index was built with {manifest['embed_model']}, config uses {config['embed_model']}.")

    wanted = ["index.tar.gz"] + ([] if args.index_only else ["corpus.tar.gz"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in wanted:
            print(f"Fetching {version}/{name} ({manifest['files'][name]['bytes'] / 1e6:.0f} MB) ...")
            source.download(f"{version}/{name}", tmp / name)
            if sha256(tmp / name) != manifest["files"][name]["sha256"]:
                sys.exit(f"✗ Checksum mismatch for {name} — not installing.")

        # Swap in the new index only after every file verified
        unpacked = tmp / "unpacked"
        _extract(tmp / "index.tar.gz", unpacked)
        index_dir = Path(config["index_path"])
        if index_dir.exists():
            shutil.rmtree(index_dir)
        shutil.move(str(unpacked / "index"), str(index_dir))
        if "corpus.tar.gz" in wanted:
            _extract(tmp / "corpus.tar.gz", Path("."))

    Path(index_dir, "VERSION.json").write_text(json.dumps({**manifest["data"], "artifact": version}, indent=1))
    print(f"✅ Installed {version} → {index_dir}/" + ("" if args.index_only else " and corpus/"))
    print("   Restart the judge to load it (on the Pi: sudo systemctl restart mtg-judge).")


if __name__ == "__main__":
    main()
