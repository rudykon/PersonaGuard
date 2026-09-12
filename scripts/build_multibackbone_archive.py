#!/usr/bin/env python3
"""Build an equal-weight archive from multiple frozen visual backbones."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.av_content_prior import FoundationFeatureArchive  # noqa: E402


DEFAULT_AST_ARCHIVE = (
    PROJECT_ROOT / "data" / "stimuli" / "features" / "foundation_av.npz"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual-archive",
        action="append",
        required=True,
        metavar="NAME:MODALITY:PATH",
        help="Repeat for each backbone, e.g. clip:clip:path and siglip:visual:path.",
    )
    parser.add_argument(
        "--visual-weight",
        action="append",
        default=None,
        metavar="NAME:WEIGHT",
        help="Optional positive pre-normalization weight for a visual backbone.",
    )
    parser.add_argument("--ast-archive", type=Path, default=DEFAULT_AST_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--delta-lags", nargs="*", type=int, default=(),
        help="Past-only embedding differences to append, e.g. 1 3.",
    )
    parser.add_argument(
        "--delta-weight", type=float, default=0.5,
    )
    parser.add_argument(
        "--normalize-ast", action="store_true",
        help="L2-normalize each AST second before joint audiovisual matching.",
    )
    parser.add_argument("--ast-weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, str, Path]:
    pieces = str(value).split(":", 2)
    if len(pieces) != 3 or not all(pieces):
        raise ValueError(
            f"Invalid --visual-archive {value!r}; expected NAME:MODALITY:PATH"
        )
    name, modality, path = pieces
    if modality not in {"clip", "visual"}:
        raise ValueError("Visual source modality must be clip or visual")
    return name, modality, Path(path)
def parse_weight(value: str) -> tuple[str, float]:
    pieces = str(value).split(":", 1)
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"Invalid --visual-weight {value!r}; expected NAME:WEIGHT")
    weight = float(pieces[1])
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("Visual backbone weights must be finite and positive")
    return pieces[0], weight




def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(
        np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8
    )
def causal_embedding_difference(values: np.ndarray, lag: int) -> np.ndarray:
    """Return a past-only embedding difference, resetting at each video."""
    matrix = np.asarray(values, dtype=np.float32)
    if int(lag) < 1:
        raise ValueError("lag must be positive")
    positions = np.maximum(np.arange(len(matrix)) - int(lag), 0)
    return matrix - matrix[positions]




def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ast_path = (
        args.ast_archive
        if args.ast_archive.is_absolute()
        else PROJECT_ROOT / args.ast_archive
    )
    delta_lags = tuple(sorted(set(int(value) for value in args.delta_lags)))
    if any(value < 1 for value in delta_lags):
        raise ValueError("delta lags must be positive")
    if float(args.delta_weight) < 0.0:
        raise ValueError("delta weight must be non-negative")
    if float(args.ast_weight) <= 0.0:
        raise ValueError("ast weight must be positive")
    sources = []
    for name, modality, path in (parse_source(value) for value in args.visual_archive):
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        sources.append((name, modality, resolved, FoundationFeatureArchive.load(resolved)))
    names = [item[0] for item in sources]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Need at least two uniquely named visual backbones")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    supplied_weights = [parse_weight(value) for value in (args.visual_weight or ())]
    supplied_names = [name for name, _ in supplied_weights]
    if len(supplied_names) != len(set(supplied_names)):
        raise ValueError("Duplicate --visual-weight name")
    unknown_weights = sorted(set(supplied_names) - set(names))
    if unknown_weights:
        raise ValueError("Weights supplied for unknown backbones: " + ", ".join(unknown_weights))
    weight_by_name = {name: 1.0 for name in names}
    weight_by_name.update(dict(supplied_weights))
    source_scales = np.asarray([weight_by_name[name] for name in names], dtype=np.float32)
    source_scales /= np.linalg.norm(source_scales)
    ast_archive = FoundationFeatureArchive.load(ast_path)
    videos = sorted(ast_archive.features_by_video)
    for name, _, _, archive in sources:
        if sorted(archive.features_by_video) != videos:
            raise ValueError(f"Video mismatch in {name}")
        if archive.annotation_lengths != ast_archive.annotation_lengths:
            raise ValueError(f"Annotation-length mismatch in {name}")
        if archive.media_sha256 != ast_archive.media_sha256:
            raise ValueError(f"Media-hash mismatch in {name}")
    ast_columns = ast_archive._columns("ast")
    ast_names = tuple(ast_archive.feature_names[index] for index in ast_columns)
    visual_names: list[str] = []
    source_columns = []
    for name, modality, _, archive in sources:
        columns = archive._columns(modality)
        source_columns.append(columns)
        visual_names.extend(
            f"visual_{name}_{position:04d}" for position in range(len(columns))
        )
        for lag in delta_lags:
            visual_names.extend(
                f"visual_{name}_delta{lag}_{position:04d}"
                for position in range(len(columns))
            )
    features = []
    video_ids = []
    timestamps = []
    media_seconds = []
    for video in videos:
        blocks = []
        expected_length = None
        for source_index, ((_, _, _, archive), columns) in enumerate(
            zip(sources, source_columns)
        ):
            scale = np.float32(source_scales[source_index])
            values = normalize_rows(
                np.asarray(archive.features_by_video[video][:, columns])
            )
            if expected_length is None:
                expected_length = len(values)
            elif len(values) != expected_length:
                raise ValueError(f"Feature length mismatch for video {video}")
            blocks.append(values * scale)
            delta_scale = (
                scale * np.float32(float(args.delta_weight) / np.sqrt(len(delta_lags)))
                if delta_lags
                else np.float32(0.0)
            )
            for lag in delta_lags:
                blocks.append(causal_embedding_difference(values, lag) * delta_scale)
        ast = np.asarray(
            ast_archive.features_by_video[video][:, ast_columns], dtype=np.float32
        )
        if len(ast) != expected_length:
            raise ValueError(f"AST length mismatch for video {video}")
        if args.normalize_ast:
            ast = normalize_rows(ast) * np.float32(args.ast_weight)
        features.append(np.concatenate([*blocks, ast], axis=1).astype(np.float32))
        video_ids.extend([video] * int(expected_length))
        timestamps.extend(range(int(expected_length)))
        media_seconds.append(int(expected_length))
    fusion_rule = (
        "L2-normalize each visual backbone per second; concatenate equal-norm levels"
    )
    if delta_lags:
        fusion_rule += "; append scaled past-only visual embedding differences"
    fusion_rule += (
        "; append L2-normalized AST" if args.normalize_ast else "; append AST unchanged"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray("merps-multibackbone-features-v1"),
        features=np.concatenate(features).astype(np.float32),
        video_ids=np.asarray(video_ids, dtype=np.int16),
        timestamps=np.asarray(timestamps, dtype=np.int16),
        feature_names=np.asarray(tuple(visual_names) + ast_names),
        record_video_ids=np.asarray(videos, dtype=np.int16),
        media_sha256=np.asarray([ast_archive.media_sha256[v] for v in videos]),
        media_seconds=np.asarray(media_seconds, dtype=np.int16),
        annotation_seconds=np.asarray(
            [ast_archive.annotation_lengths[v] for v in videos], dtype=np.int16
        ),
        visual_backbones=np.asarray(names),
        visual_source_paths=np.asarray([str(item[2]) for item in sources]),
        visual_source_sha256=np.asarray([file_sha256(item[2]) for item in sources]),
        ast_source_path=np.asarray(str(ast_path)),
        visual_backbone_weights=np.asarray([weight_by_name[name] for name in names]),
        ast_source_sha256=np.asarray(file_sha256(ast_path)),
        fusion_rule=np.asarray(fusion_rule),
        visual_delta_lags=np.asarray(delta_lags, dtype=np.int16),
        visual_delta_weight=np.asarray(float(args.delta_weight), dtype=np.float32),
        ast_normalized=np.asarray(bool(args.normalize_ast)),
        ast_weight=np.asarray(float(args.ast_weight), dtype=np.float32),
    )
    manifest = {
        "schema_version": "merps-multibackbone-features-v1",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "visual_backbones": names,
        "fusion_rule": fusion_rule,
        "visual_delta_lags": list(delta_lags),
        "visual_delta_weight": float(args.delta_weight),
        "visual_backbone_weights": {name: weight_by_name[name] for name in names},
        "sources": {
            name: {"path": str(path), "sha256": file_sha256(path), "modality": modality}
            for name, modality, path, _ in sources
        },
        "ast_normalized": bool(args.normalize_ast),
        "ast_weight": float(args.ast_weight),
        "ast_source": {"path": str(ast_path), "sha256": file_sha256(ast_path)},
        "media_identity_consistent": True,
        "videos": videos,
    }
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
