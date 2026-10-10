#!/usr/bin/env python3
"""Publish the built index + a corpus snapshot as an immutable, versioned artifact.

  python scripts/publish_data.py                     # or: make publish
  MTG_JUDGE_ARTIFACTS=~/Dropbox/mtg-judge-artifacts python scripts/publish_data.py

Layout (same for every backend):
  <root>/latest.json                 {"version": ...}
  <root>/<version>/manifest.json      data versions (version = date, CR, cards, content hash), embed model, file sizes + sha256
  <root>/<version>/index.tar.gz       index/ (Chroma, BM25, parents, card names)
  <root>/<version>/corpus.tar.gz      raw sources: CR, MTR/IPG/JAR, Scryfall cards +
                                      rulings, versions.json, metagame decklists

Derived corpus files (chunks.jsonl, parents.jsonl) are left out — `make corpus`'s
parse step rebuilds them from the snapshot in about a minute.

Backends: a local folder (e.g. one synced by the Dropbox app). The layout is plain
files so an S3-compatible backend (Cloudflare R2) can be added without changing it.
The artifacts contain Wizards' rules text verbatim — keep the storage private.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, ".")
from src.config import load_config

DEFAULT_ROOT = "~/Dropbox/mtg-judge-artifacts"
CORPUS_FILES = [
    "cr.txt", "mtr.jsonl", "ipg.jsonl", "jar.jsonl",
    "scryfall_oracle.jsonl.gz", "scryfall_rulings.jsonl.gz", "versions.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tar(out: Path, members: list[tuple[Path, str]]) -> None:
    """Deterministic .tar.gz: same files → same bytes → same sha256 (gzip header mtime = 0)."""
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for src, arcname in members:
            tar.add(src, arcname=arcname)  # directories recurse in sorted order


class LocalBackend:
    """A plain folder — e.g. ~/Dropbox/mtg-judge-artifacts, uploaded by the Dropbox app."""

    def __init__(self, root: str):
        self.root = Path(os.path.expanduser(root))

    def exists(self, version: str) -> bool:
        return (self.root / version / "manifest.json").exists()

    def put_version(self, version: str, files: list[Path]) -> None:
        dest = self.root / version
        dest.mkdir(parents=True, exist_ok=False)
        for f in files:
            shutil.copy2(f, dest / f.name)

    def set_latest(self, version: str) -> None:
        (self.root / "latest.json").write_text(json.dumps({"version": version}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config")
    parser.add_argument("--dest", default=os.environ.get("MTG_JUDGE_ARTIFACTS", DEFAULT_ROOT))
    args = parser.parse_args()
    config = load_config(args.config)

    index_dir, corpus = Path(config["index_path"]), Path("corpus")
    for required in [index_dir / "chroma", index_dir / "parents.jsonl", corpus / "versions.json"]:
        if not required.exists():
            sys.exit(f"✗ {required} missing — run `make update` (or corpus + index) first.")

    versions = json.loads((corpus / "versions.json").read_text())
    backend = LocalBackend(args.dest)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("Packing index + corpus ...")
        index_tar, corpus_tar = tmp / "index.tar.gz", tmp / "corpus.tar.gz"
        _tar(index_tar, [(p, f"index/{p.name}") for p in sorted(index_dir.iterdir()) if p.name != "VERSION.json"])
        corpus_members = [(corpus / f, f"corpus/{f}") for f in CORPUS_FILES if (corpus / f).exists()]
        if (corpus / "metagame").exists():
            corpus_members.append((corpus / "metagame", "corpus/metagame"))
        _tar(corpus_tar, corpus_members)
        files = {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in (index_tar, corpus_tar)}

        # Name = what's inside + a content hash: same data → same version (skipped),
        # new data (e.g. more metagame events the same day) → a new version.
        content = hashlib.sha256("".join(f["sha256"] for f in files.values()).encode()).hexdigest()[:8]
        version = (f"{date.today()}_cr{versions.get('cr', 'unknown')}"
                   f"_cards{versions.get('scryfall_cards', '')[:10]}_{content}")
        if backend.exists(version):
            print(f"✓ {version} is already published (identical content) — nothing to do.")
            return

        manifest = {
            "version": version,
            "data": versions,
            "embed_model": config["embed_model"],
            "doc_prefix_scheme": "nomic search_document/search_query",
            "files": files,
        }
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

        backend.put_version(version, [index_tar, corpus_tar, manifest_path])
        backend.set_latest(version)

    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"✅ Published {version} ({total / 1e6:.0f} MB) → {backend.root / version}")
    for name, f in manifest["files"].items():
        print(f"   {name:16} {f['bytes'] / 1e6:7.1f} MB  sha256 {f['sha256'][:16]}…")
    print(f"   latest.json → {version}")


if __name__ == "__main__":
    main()
