#!/usr/bin/env python3
"""Extract resumable frozen visual embeddings and reuse the audited AST stream."""

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

from merps.av_content_prior import FoundationFeatureArchive  # noqa: E402
from merps.content_prior import load_stimulus_manifest  # noqa: E402
from merps.foundation_features import (  # noqa: E402
    TIMESTAMP_CONVENTION,
    decode_visual_seconds,
    model_revision,
)
from merps.stimulus_features import audit_stimuli  # noqa: E402
from merps.visual_backbone_features import (  # noqa: E402
    BACKBONE_MODEL_IDS,
    SCHEMA_VERSION,
    encode_visual_frames,
    load_visual_backbone,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "data" / "stimuli" / "media"
DEFAULT_AUDIO_ARCHIVE = (
    PROJECT_ROOT / "data" / "stimuli" / "features" / "foundation_av.npz"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--audio-archive", type=Path, default=DEFAULT_AUDIO_ARCHIVE)
    parser.add_argument("--backbone", choices=tuple(BACKBONE_MODEL_IDS), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--video-ids", nargs="+", type=int)
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
    model_id: str,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                int(payload["video_id"]) == int(video_id)
                and str(payload["media_sha256"]) == str(media_sha256)
                and str(payload["visual_model_id"]) == str(model_id)
                and payload["visual"].ndim == 2
                and bool(np.isfinite(payload["visual"]).all())
            )
    except (OSError, ValueError, KeyError):
        return False


def finalize(
    output: Path,
    shard_dir: Path,
    records,
    audio_archive_path: Path,
    *,
    backbone: str,
    model_id: str,
    model_revision_value: str,
) -> bool:
    paths = [shard_dir / f"video_{record.video_id:02d}.npz" for record in records]
    if not all(path.exists() for path in paths):
        return False
    audio_archive = FoundationFeatureArchive.load(audio_archive_path)
    with np.load(audio_archive_path, allow_pickle=False) as source:
        ast_model_id = str(source["ast_model_id"])
        ast_model_revision = str(source["ast_model_revision"])
        audio_window_seconds = int(source["audio_window_seconds"])
        causal_audio = bool(source["causal_audio"])
    ast_columns = audio_archive._columns("ast")
    ast_names = tuple(audio_archive.feature_names[index] for index in ast_columns)
    features = []
    video_ids = []
    timestamps = []
    hashes = []
    media_seconds = []
    annotation_seconds = []
    visual_names: tuple[str, ...] | None = None
    for record, path in zip(records, paths):
        with np.load(path, allow_pickle=False) as payload:
            visual = np.asarray(payload["visual"], dtype=np.float32)
            if int(payload["video_id"]) != int(record.video_id):
                raise RuntimeError(f"Video ID mismatch in {path}")
            hashes.append(str(payload["media_sha256"]))
        full = np.asarray(
            audio_archive.features_by_video[int(record.video_id)], dtype=np.float32
        )
        ast = full[:, ast_columns]
        if len(visual) != len(ast):
            raise RuntimeError(
                f"Visual/AST length mismatch for video {record.video_id}: "
                f"{len(visual)} != {len(ast)}"
            )
        if visual_names is None:
            visual_names = tuple(
                f"visual_{index:04d}" for index in range(visual.shape[1])
            )
        elif len(visual_names) != visual.shape[1]:
            raise RuntimeError("Visual shard dimensions differ")
        features.append(np.concatenate([visual, ast], axis=1).astype(np.float32))
        video_ids.extend([record.video_id] * len(visual))
        timestamps.extend(range(len(visual)))
        media_seconds.append(len(visual))
        annotation_seconds.append(int(record.annotation_seconds))
    assert visual_names is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(SCHEMA_VERSION),
        features=np.concatenate(features).astype(np.float32),
        video_ids=np.asarray(video_ids, dtype=np.int16),
        timestamps=np.asarray(timestamps, dtype=np.int16),
        feature_names=np.asarray(visual_names + ast_names),
        record_video_ids=np.asarray(
            [record.video_id for record in records], dtype=np.int16
        ),
        media_sha256=np.asarray(hashes),
        media_seconds=np.asarray(media_seconds, dtype=np.int16),
        annotation_seconds=np.asarray(annotation_seconds, dtype=np.int16),
        visual_backbone=np.asarray(backbone),
        visual_model_id=np.asarray(model_id),
        visual_model_revision=np.asarray(model_revision_value),
        ast_model_id=np.asarray(ast_model_id),
        ast_model_revision=np.asarray(ast_model_revision),
        audio_window_seconds=np.asarray(audio_window_seconds, dtype=np.int16),
        causal_audio=np.asarray(causal_audio),
        timestamp_convention=np.asarray(TIMESTAMP_CONVENTION),
        source_audio_archive_sha256=np.asarray(file_sha256(audio_archive_path)),
    )
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    model_id = str(args.model_id or BACKBONE_MODEL_IDS[args.backbone])
    output = args.output or (
        PROJECT_ROOT
        / "data"
        / "stimuli"
        / "features"
        / f"foundation_{args.backbone}_ast.npz"
    )
    records = load_stimulus_manifest(args.manifest)
    selected_ids = (
        set(int(value) for value in args.video_ids)
        if args.video_ids
        else {int(record.video_id) for record in records}
    )
    unknown = selected_ids.difference(int(record.video_id) for record in records)
    if unknown:
        raise ValueError(f"Unknown video IDs: {sorted(unknown)}")
    audit = audit_stimuli(records, args.media_dir, ffprobe=args.ffprobe)
    if not all(item["status"] == "identity_verified" for item in audit):
        raise RuntimeError("Visual extraction requires all media identities verified")
    audit_by_id = {int(item["video_id"]): item for item in audit}

    import torch
    import transformers

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    processor, model = load_visual_backbone(
        model_id,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    revision = model_revision(model, model_id)
    shard_dir = output.parent / f"{output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for position, record in enumerate(records, start=1):
        if int(record.video_id) not in selected_ids:
            continue
        audit_item = audit_by_id[int(record.video_id)]
        shard_path = shard_dir / f"video_{record.video_id:02d}.npz"
        if (
            shard_valid(
                shard_path,
                video_id=record.video_id,
                media_sha256=str(audit_item["sha256"]),
                model_id=model_id,
            )
            and not args.overwrite_shards
        ):
            print(f"[{position:02d}/15] reuse video {record.video_id:02d}", flush=True)
            continue
        print(f"[{position:02d}/15] encode video {record.video_id:02d}", flush=True)
        seconds = max(1, int(np.ceil(float(audit_item["duration_seconds"]))))
        frames = decode_visual_seconds(
            Path(str(audit_item["path"])),
            seconds=seconds,
            ffmpeg=args.ffmpeg,
        )
        visual = encode_visual_frames(
            frames,
            processor,
            model,
            device=args.device,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            shard_path,
            schema_version=np.asarray(SCHEMA_VERSION),
            video_id=np.asarray(record.video_id, dtype=np.int16),
            visual=visual,
            media_sha256=np.asarray(str(audit_item["sha256"])),
            media_duration_seconds=np.asarray(
                float(audit_item["duration_seconds"]), dtype=np.float64
            ),
            annotation_seconds=np.asarray(record.annotation_seconds, dtype=np.int16),
            visual_backbone=np.asarray(args.backbone),
            visual_model_id=np.asarray(model_id),
            visual_model_revision=np.asarray(revision),
        )
    complete = False
    if not args.no_finalize:
        complete = finalize(
            output,
            shard_dir,
            records,
            args.audio_archive,
            backbone=args.backbone,
            model_id=model_id,
            model_revision_value=revision,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "partial",
        "completed_at_utc": utc_now(),
        "output": str(output),
        "output_sha256": file_sha256(output) if complete else None,
        "media_identity_verified": True,
        "videos_encoded": sorted(selected_ids),
        "visual_backbone": args.backbone,
        "visual_model_id": model_id,
        "visual_model_revision": revision,
        "audio_archive": str(args.audio_archive),
        "audio_archive_sha256": file_sha256(args.audio_archive),
        "timestamp_convention": TIMESTAMP_CONVENTION,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
