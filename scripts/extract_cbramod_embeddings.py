#!/usr/bin/env python3
"""Extract frozen CBraMod patch and pooled embeddings from cached raw EEG."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.cbramod import (
    cbramod_patch_tokens,
    load_pretrained_cbramod,
    pool_cbramod_features,
)
from merps.innovation.data import SignalWindowStore, load_innovation_index


class EegWindowDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        subjects: np.ndarray,
        videos: np.ndarray,
        timestamps: np.ndarray,
        global_indices: np.ndarray,
        *,
        causal: bool = False,
    ):
        self.cache_dir = cache_dir
        self.subjects = subjects
        self.videos = videos
        self.timestamps = timestamps
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.causal = bool(causal)
        self._store: SignalWindowStore | None = None

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, local_index: int):
        if self._store is None:
            self._store = SignalWindowStore(self.cache_dir)
        row = int(self.global_indices[local_index])
        eeg = self._store.eeg_window(
            str(self.subjects[row]),
            int(self.videos[row]),
            int(self.timestamps[row]),
            seconds=4,
            causal=self.causal,
        )
        return torch.from_numpy(eeg), row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "MER_PS_trainval",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "innovation_cache",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints"
        / "innovation"
        / "pretrained_weights.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "innovation_cache" / "cbramod",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use only t-3,t-2,t-1,t; required for a real-time claim.",
    )
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(args: argparse.Namespace) -> None:
    index = load_innovation_index(args.data_root)
    total = len(index.targets)
    pooled_path = args.output_dir / "pooled.npy"
    tokens_path = args.output_dir / "patch_tokens.npy"
    pooled = np.lib.format.open_memmap(
        pooled_path, mode="w+", dtype=np.float16, shape=(total, 400)
    )
    tokens = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.float16, shape=(total, 4, 200)
    )
    coverage = np.zeros(total, dtype=np.int8)
    for shard in range(args.num_shards):
        prefix = args.output_dir / f"shard_{shard:02d}"
        rows = np.load(prefix.with_name(prefix.name + "_indices.npy"), allow_pickle=False)
        shard_pooled = np.load(
            prefix.with_name(prefix.name + "_pooled.npy"), allow_pickle=False
        )
        shard_tokens = np.load(
            prefix.with_name(prefix.name + "_tokens.npy"), allow_pickle=False
        )
        if len(rows) != len(shard_pooled) or len(rows) != len(shard_tokens):
            raise ValueError(f"Shard {shard} has inconsistent lengths")
        pooled[rows] = shard_pooled
        tokens[rows] = shard_tokens
        coverage[rows] += 1
    pooled.flush()
    tokens.flush()
    if not np.all(coverage == 1):
        missing = int(np.sum(coverage == 0))
        duplicate = int(np.sum(coverage > 1))
        raise RuntimeError(
            f"Embedding coverage is invalid: missing={missing} duplicate={duplicate}"
        )
    manifest = {
        "status": "complete",
        "samples": total,
        "patch_tokens": [total, 4, 200],
        "pooled": [total, 400],
        "dtype": "float16",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "source": "weighting666/CBraMod pretrained_weights.pth",
        "license": "MIT",
        "num_shards": int(args.num_shards),
        "causal": bool(args.causal),
        "temporal_support": (
            "t-3,t-2,t-1,t with trial-start edge replication"
            if args.causal
            else "centered four-second window containing future samples"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def extract(args: argparse.Namespace) -> None:
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num-shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = load_innovation_index(args.data_root)
    rows = np.arange(args.shard_id, len(index.targets), args.num_shards, dtype=np.int64)
    dataset = EegWindowDataset(
        args.cache_dir,
        index.subjects,
        index.videos,
        index.timestamps,
        rows,
        causal=args.causal,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_pretrained_cbramod(args.checkpoint).to(device).eval()
    pooled_parts = []
    token_parts = []
    row_parts = []
    start = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for eeg, global_rows in loader:
            eeg = eeg.to(device, non_blocking=True, dtype=torch.float32)
            features = model(eeg)
            pooled_parts.append(
                pool_cbramod_features(features).cpu().numpy().astype(np.float16)
            )
            token_parts.append(
                cbramod_patch_tokens(features).cpu().numpy().astype(np.float16)
            )
            row_parts.append(global_rows.numpy().astype(np.int64))
            completed += len(eeg)
            if completed % max(args.batch_size * 20, 1) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"shard={args.shard_id} completed={completed}/{len(dataset)} "
                    f"samples_per_second={completed / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
    prefix = args.output_dir / f"shard_{args.shard_id:02d}"
    np.save(
        prefix.with_name(prefix.name + "_indices.npy"),
        np.concatenate(row_parts),
    )
    np.save(
        prefix.with_name(prefix.name + "_pooled.npy"),
        np.concatenate(pooled_parts),
    )
    np.save(
        prefix.with_name(prefix.name + "_tokens.npy"),
        np.concatenate(token_parts),
    )
    elapsed = time.perf_counter() - start
    print(
        f"shard={args.shard_id} complete samples={len(dataset)} "
        f"seconds={elapsed:.2f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        extract(args)


if __name__ == "__main__":
    main()
