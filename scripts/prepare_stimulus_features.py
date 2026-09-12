#!/usr/bin/env python3
"""Audit exact REFED stimuli and extract timestamp-aligned AV features."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.content_prior import load_stimulus_manifest  # noqa: E402
from merps.stimulus_features import (  # noqa: E402
    SCHEMA_VERSION,
    VideoFeatureRecord,
    audit_stimuli,
    extract_low_level_features,
    pack_feature_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv",
    )
    parser.add_argument(
        "--media-dir", type=Path, default=PROJECT_ROOT / "data" / "stimuli" / "media"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "stimuli" / "features" / "low_level_av.npz",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "stimuli" / "stimulus_audit.json",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--allow-unverified-media",
        action="store_true",
        help="Extract duration-matched files without an authoritative SHA-256; output remains development-only.",
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def portable_path(path: Path) -> str:
    """Return a repository-relative path without exposing a local home directory."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def main() -> None:
    args = parse_args()
    records = load_stimulus_manifest(args.manifest)
    audit = audit_stimuli(records, args.media_dir, ffprobe=args.ffprobe)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": portable_path(args.manifest),
        "media_dir": portable_path(args.media_dir),
        "files_expected": len(records),
        "files_present": sum(bool(item["present"]) for item in audit),
        "identity_verified": sum(bool(item["identity_verified"]) for item in audit),
        "duration_only_unverified": sum(
            item["status"] == "duration_only_unverified" for item in audit
        ),
        "formal_feature_status": (
            "ready" if all(item["status"] == "identity_verified" for item in audit) else "blocked"
        ),
        "development_feature_status": (
            "ready"
            if all(
                item["status"] in {"identity_verified", "duration_only_unverified"}
                for item in audit
            )
            else "blocked"
        ),
        "records": [
            {**item, "path": portable_path(Path(str(item["path"])))}
            for item in audit
        ],
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Stimuli present={summary['files_present']}/{summary['files_expected']} "
        f"identity-verified={summary['identity_verified']}/{summary['files_expected']}",
        flush=True,
    )
    if args.audit_only:
        return
    accepted = {"identity_verified"}
    if args.allow_unverified_media:
        accepted.add("duration_only_unverified")
    selected = [item for item in audit if item["status"] in accepted]
    if args.require_all and len(selected) != len(records):
        raise RuntimeError(
            "Not all 15 stimuli passed the requested identity policy; see stimulus_audit.json"
        )
    if not selected:
        raise RuntimeError(
            "No stimuli passed identity policy. Obtain the exact edits or use "
            "--allow-unverified-media only for development checks."
        )
    by_id = {record.video_id: record for record in records}
    feature_records = []
    for position, item in enumerate(selected, start=1):
        record = by_id[int(item["video_id"])]
        print(f"Extracting video {record.video_id:02d} ({position}/{len(selected)})", flush=True)
        features, names = extract_low_level_features(
            Path(item["path"]),
            annotation_seconds=record.annotation_seconds,
            has_audio=bool(item.get("has_audio", False)),
            ffmpeg=args.ffmpeg,
        )
        feature_records.append(
            VideoFeatureRecord(
                video_id=record.video_id,
                features=features,
                feature_names=names,
                media_sha256=str(item["sha256"]),
                identity_status=str(item["status"]),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **pack_feature_records(feature_records))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
