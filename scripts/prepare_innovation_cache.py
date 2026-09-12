#!/usr/bin/env python3
"""Build the raw multi-timescale cache used by CLaM-R experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import (
    EEG_RESAMPLE_METHODS,
    build_handcrafted_eeg_pool,
    build_signal_cache,
    load_innovation_index,
)


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
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs for a smoke cache, for example test_1 test_2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--eeg-resampling",
        choices=EEG_RESAMPLE_METHODS,
        default="block_average",
        help="EEG 1000-to-200-Hz sensitivity path.",
    )
    parser.add_argument(
        "--build-handcrafted-eeg",
        action="store_true",
        help="Rebuild the pooled handcrafted EEG block from this signal cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_signal_cache(
        args.data_root,
        args.cache_dir,
        subjects=args.subjects,
        overwrite=args.overwrite,
        verbose=True,
        eeg_resample_method=args.eeg_resampling,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.build_handcrafted_eeg:
        if args.subjects is not None:
            raise ValueError(
                "--build-handcrafted-eeg currently requires the complete subject set"
            )
        index = load_innovation_index(args.data_root)
        build_handcrafted_eeg_pool(
            index,
            args.cache_dir,
            overwrite=args.overwrite,
            verbose=True,
        )


if __name__ == "__main__":
    main()
