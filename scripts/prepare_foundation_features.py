#!/usr/bin/env python3
"""Extract resumable, identity-audited CLIP and causal AST stimulus features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.content_prior import load_stimulus_manifest  # noqa: E402
from merps.foundation_features import (  # noqa: E402
    AST_MODEL_ID,
    CLIP_MODEL_ID,
    SCHEMA_VERSION,
    TIMESTAMP_CONVENTION,
    extract_foundation_features,
    load_foundation_models,
    model_revision,
)
from merps.stimulus_features import audit_stimuli  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "data" / "stimuli" / "media"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stimuli" / "features" / "foundation_av.npz"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-model", default=CLIP_MODEL_ID)
    parser.add_argument("--ast-model", default=AST_MODEL_ID)
    parser.add_argument("--visual-batch-size", type=int, default=64)
    parser.add_argument("--audio-batch-size", type=int, default=8)
    parser.add_argument("--audio-window-seconds", type=int, default=10)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument(
        "--video-ids",
        nargs="+",
        type=int,
        help="Optional subset for sharded parallel extraction; finalization waits for all 15.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_valid(
    path: Path,
    *,
    video_id: int,
    media_sha256: str,
    clip_model: str,
    ast_model: str,
    audio_window_seconds: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                int(payload["video_id"]) == int(video_id)
                and str(payload["media_sha256"]) == str(media_sha256)
                and str(payload["clip_model_id"]) == str(clip_model)
                and str(payload["ast_model_id"]) == str(ast_model)
                and int(payload["audio_window_seconds"]) == int(audio_window_seconds)
                and payload["features"].ndim == 2
                and bool(np.isfinite(payload["features"]).all())
            )
    except (OSError, ValueError, KeyError):
        return False


def finalize(
    output: Path,
    shard_dir: Path,
    records,
    *,
    clip_model_id: str,
    clip_revision: str,
    ast_model_id: str,
    ast_revision: str,
    audio_window_seconds: int,
) -> bool:
    paths = [shard_dir / f"video_{record.video_id:02d}.npz" for record in records]
    if not all(path.exists() for path in paths):
        return False
    features = []
    video_ids = []
    timestamps = []
    names = None
    media_sha256 = []
    media_seconds = []
    annotation_seconds = []
    for record, path in zip(records, paths):
        with np.load(path, allow_pickle=False) as payload:
            local_names = tuple(str(value) for value in payload["feature_names"])
            if names is None:
                names = local_names
            if local_names != names:
                raise RuntimeError(f"Feature schema mismatch in {path}")
            values = np.asarray(payload["features"], dtype=np.float32)
            if int(payload["video_id"]) != int(record.video_id):
                raise RuntimeError(f"Video ID mismatch in {path}")
            features.append(values)
            video_ids.extend([record.video_id] * len(values))
            timestamps.extend(range(len(values)))
            media_sha256.append(str(payload["media_sha256"]))
            media_seconds.append(len(values))
            annotation_seconds.append(int(record.annotation_seconds))
    assert names is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(SCHEMA_VERSION),
        features=np.concatenate(features).astype(np.float32),
        video_ids=np.asarray(video_ids, dtype=np.int16),
        timestamps=np.asarray(timestamps, dtype=np.int16),
        feature_names=np.asarray(names),
        record_video_ids=np.asarray(
            [record.video_id for record in records], dtype=np.int16
        ),
        media_sha256=np.asarray(media_sha256),
        media_seconds=np.asarray(media_seconds, dtype=np.int16),
        annotation_seconds=np.asarray(annotation_seconds, dtype=np.int16),
        clip_model_id=np.asarray(clip_model_id),
        clip_model_revision=np.asarray(clip_revision),
        ast_model_id=np.asarray(ast_model_id),
        ast_model_revision=np.asarray(ast_revision),
        audio_window_seconds=np.asarray(audio_window_seconds, dtype=np.int16),
        causal_audio=np.asarray(True),
        timestamp_convention=np.asarray(TIMESTAMP_CONVENTION),
    )
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    records = load_stimulus_manifest(args.manifest)
    selected_ids = (
        set(int(value) for value in args.video_ids)
        if args.video_ids
        else set(record.video_id for record in records)
    )
    unknown = selected_ids.difference(record.video_id for record in records)
    if unknown:
        raise ValueError(f"Unknown video IDs: {sorted(unknown)}")
    audit = audit_stimuli(records, args.media_dir, ffprobe=args.ffprobe)
    if not all(item["status"] == "identity_verified" for item in audit):
        failed = [
            (item["video_id"], item["status"])
            for item in audit
            if item["status"] != "identity_verified"
        ]
        raise RuntimeError(f"Foundation extraction requires verified media: {failed}")
    audit_by_id = {int(item["video_id"]): item for item in audit}

    import torch
    import transformers

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    clip_processor, clip_model, ast_processor, ast_model = load_foundation_models(
        device=args.device,
        clip_model_id=args.clip_model,
        ast_model_id=args.ast_model,
        local_files_only=args.local_files_only,
    )
    clip_revision = model_revision(clip_model, args.clip_model)
    ast_revision = model_revision(ast_model, args.ast_model)

    shard_dir = args.output.parent / f"{args.output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for position, record in enumerate(records, start=1):
        if record.video_id not in selected_ids:
            continue
        audit_item = audit_by_id[record.video_id]
        shard_path = shard_dir / f"video_{record.video_id:02d}.npz"
        valid = shard_valid(
            shard_path,
            video_id=record.video_id,
            media_sha256=str(audit_item["sha256"]),
            clip_model=args.clip_model,
            ast_model=args.ast_model,
            audio_window_seconds=args.audio_window_seconds,
        )
        if valid and not args.overwrite_shards:
            print(f"[{position:02d}/15] reuse video {record.video_id:02d}", flush=True)
            continue
        print(f"[{position:02d}/15] encode video {record.video_id:02d}", flush=True)
        values, names = extract_foundation_features(
            Path(str(audit_item["path"])),
            media_duration_seconds=float(audit_item["duration_seconds"]),
            clip_processor=clip_processor,
            clip_model=clip_model,
            ast_processor=ast_processor,
            ast_model=ast_model,
            device=args.device,
            visual_batch_size=args.visual_batch_size,
            audio_batch_size=args.audio_batch_size,
            audio_window_seconds=args.audio_window_seconds,
            ffmpeg=args.ffmpeg,
        )
        np.savez_compressed(
            shard_path,
            schema_version=np.asarray(SCHEMA_VERSION),
            video_id=np.asarray(record.video_id, dtype=np.int16),
            features=values,
            feature_names=np.asarray(names),
            media_sha256=np.asarray(str(audit_item["sha256"])),
            media_duration_seconds=np.asarray(
                float(audit_item["duration_seconds"]), dtype=np.float64
            ),
            annotation_seconds=np.asarray(record.annotation_seconds, dtype=np.int16),
            clip_model_id=np.asarray(args.clip_model),
            clip_model_revision=np.asarray(clip_revision),
            ast_model_id=np.asarray(args.ast_model),
            ast_model_revision=np.asarray(ast_revision),
            audio_window_seconds=np.asarray(
                args.audio_window_seconds, dtype=np.int16
            ),
        )

    complete = False
    if not args.no_finalize:
        complete = finalize(
            args.output,
            shard_dir,
            records,
            clip_model_id=args.clip_model,
            clip_revision=clip_revision,
            ast_model_id=args.ast_model,
            ast_revision=ast_revision,
            audio_window_seconds=args.audio_window_seconds,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "partial",
        "completed_at_utc": utc_now(),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output) if complete else None,
        "media_identity_verified": True,
        "videos_encoded": sorted(selected_ids),
        "clip_model_id": args.clip_model,
        "clip_model_revision": clip_revision,
        "ast_model_id": args.ast_model,
        "ast_model_revision": ast_revision,
        "audio_window_seconds": int(args.audio_window_seconds),
        "causal_audio": True,
        "timestamp_convention": TIMESTAMP_CONVENTION,
        "offset_support_seconds": [-3, -2, -1, 0, 1, 2, 3],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
