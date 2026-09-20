#!/usr/bin/env python3
"""Build and validate the deterministic anonymous CHI supplementary archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUTPUT = PAPER / "submission" / "chi2027_anonymous_supplement.zip"
ARCHIVE_ROOT = "chi2027_anonymous_supplement"
FIXED_TIMESTAMP = (2026, 8, 14, 0, 0, 0)

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
RESTRICTED_SUFFIXES = {
    ".avi",
    ".edf",
    ".mat",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".set",
    ".snirf",
}
RESTRICTED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "data",
    "submission",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:\\)")

DIRECT_FILES = (
    "requirements.txt",
    "paper/accessibility_patches.tex",
    "configs/protocol/adjacent_frameworks.json",
    "results/evidence_traceability.json",
    "configs/protocol/external_reuse_cases.json",
    "configs/protocol/protocol_replay_cases.json",
    "configs/protocol/protocol_rules.json",
    "paper/references.bib",
    "results/revision6_source.json",
    "scripts/figures/dense_trajectory_examples.json",
    "paper/supplementary_information.pdf",
    "paper/supplementary_information.tex",
    "paper/word_count_report.txt",
)

GLOBS = (
    ("src", "*.py"),
    ("scripts", "*.py"),
    ("tests", "test_*.py"),
    ("configs", "*.csv"),
    ("configs", "*.json"),
    ("configs", "*.yaml"),
    ("configs", "*.yml"),
    ("results", "*.json"),
    ("paper/generated", "*.tex"),
    ("paper/figures", "*.svg"),
    ("paper/figures", "*_embed.pdf"),
    ("scripts/figures", "source_data_*.csv"),
)


# Superseded known-video fixed-fusion implementations have been removed.
# Keep this deny-list as a regression guard so accidentally restored files
# cannot enter the anonymous supplement through the broad source/test globs.
LEGACY_KNOWN_VIDEO_FILES = frozenset(
    {
        "docs/ablation.md",
        "docs/dataset.md",
        "docs/method.md",
        "docs/nested_neural.md",
        "docs/p0_p2.md",
        "docs/history/README.md",
        "docs/history/ablation.md",
        "docs/history/dataset.md",
        "docs/history/method.md",
        "docs/history/nested_neural.md",
        "docs/history/p0_p2.md",
        "paper/figures/make_external_figures.py",
        "paper/figures/make_figures.py",
        "paper/figures/source_data_components.csv",
        "scripts/analyze_algorithm_bottlenecks.py",
        "scripts/evaluate.py",
        "scripts/evaluate_external.py",
        "scripts/export_model_bundle.py",
        "scripts/run_known_video_physio_residual.py",
        "scripts/run_p0_p2_experiments.py",
        "scripts/train_cv.py",
        "scripts/train_nested_neural.py",
        "scripts/train_split.py",
        "scripts/validate_model_bundle.py",
        "src/merps/calibration.py",
        "src/merps/inference.py",
        "src/merps/known_video_residual.py",
        "src/merps/model.py",
        "src/merps/prior.py",
        "tests/test_calibration.py",
        "tests/test_evaluate_external.py",
        "tests/test_export_model_bundle.py",
        "tests/test_known_video_residual.py",
        "tests/test_nested_neural.py",
        "tests/test_p0_p2_experiments.py",
    }
)
LEGACY_KNOWN_VIDEO_PREFIXES = (
    "paper/figures/source_data_external_",
)
LEGACY_RESULT_MARKERS = (
    b"27." + b"7246",
    b"29." + b"0146",
    b"47." + b"3509",
    b"[0.99," + b" 0.92]",
)


def is_legacy_known_video_path(relative_path: Path | PurePosixPath | str) -> bool:
    relative = PurePosixPath(str(relative_path).replace("\\", "/")).as_posix()
    return relative in LEGACY_KNOWN_VIDEO_FILES or any(
        relative.startswith(prefix) for prefix in LEGACY_KNOWN_VIDEO_PREFIXES
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def archive_name(relative_path: Path) -> str:
    if relative_path == Path("paper/supplementary_information.pdf"):
        return f"{ARCHIVE_ROOT}/supplementary_information.pdf"
    return f"{ARCHIVE_ROOT}/{relative_path.as_posix()}"


def collect_files() -> list[Path]:
    selected: set[Path] = set()
    for relative in DIRECT_FILES:
        candidate = ROOT / relative
        if candidate.is_file():
            selected.add(candidate)
    for base, pattern in GLOBS:
        directory = ROOT / base
        if directory.is_dir():
            selected.update(path for path in directory.rglob(pattern) if path.is_file())

    clean: list[Path] = []
    for path in selected:
        relative = path.relative_to(ROOT)
        # Chinese counterparts are local reading aids, not anonymous CHI
        # submission dependencies. Keep the formal supplement archive English-only.
        if path.name.endswith("_zh.tex"):
            continue
        if is_legacy_known_video_path(relative):
            continue
        if path.suffix.lower() in RESTRICTED_SUFFIXES:
            raise ValueError(f"Restricted binary selected: {relative}")
        if any(part in RESTRICTED_PARTS for part in relative.parts):
            raise ValueError(f"Restricted directory selected: {relative}")
        if path.name.endswith(".orig") or path.name.endswith("~"):
            continue
        clean.append(path)
    return sorted(clean, key=lambda item: item.relative_to(ROOT).as_posix())


def private_markers() -> tuple[str, ...]:
    home = Path.home()
    user = home.name
    return (
        str(ROOT),
        str(home),
        f"/Users/{user}",
        "C:" + "\\Users\\" + user,
        "file" + "://",
    )


def assert_safe_member(name: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    if not pure.parts or pure.parts[0] != ARCHIVE_ROOT:
        raise ValueError(f"Member outside archive root: {name}")
    if any(part in RESTRICTED_PARTS for part in pure.parts[1:]):
        raise ValueError(f"Restricted path in archive: {name}")
    if pure.suffix.lower() in RESTRICTED_SUFFIXES:
        raise ValueError(f"Restricted binary in archive: {name}")
    relative = PurePosixPath(*pure.parts[1:])
    if is_legacy_known_video_path(relative):
        raise ValueError(f"Superseded known-video material in archive: {name}")


def assert_anonymous_text(name: str, payload: bytes) -> None:
    for marker in private_markers():
        encoded = marker.encode("utf-8")
        if encoded and encoded in payload:
            raise ValueError(f"Private path marker {marker!r} in {name}")
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        return
    text = payload.decode("utf-8", errors="strict")
    if WINDOWS_ABSOLUTE_PATH.search(text):
        raise ValueError(f"Windows absolute path in {name}")


def assert_no_legacy_results(name: str, payload: bytes) -> None:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        return
    for marker in LEGACY_RESULT_MARKERS:
        if marker in payload:
            raise ValueError(
                f"Superseded known-video result marker {marker!r} in {name}"
            )


def readme_payload() -> bytes:
    text = """# Anonymous supplementary materials

This archive accompanies an anonymized CHI submission. It contains the
supplementary-information PDF, portable numerical provenance, aggregate source
data for figures, analysis and validation code, configuration files, and tests.
The sole trial-level exception is Figure 5: three author-authorized anonymous
examples and their plotted coordinates. Original identities are not included.

The archive intentionally excludes raw or participant-level EEG/fNIRS arrays,
complete per-participant predictions, stimulus media, extracted gated features,
checkpoints, credentials, repository metadata, and machine-specific paths.
It also excludes superseded known-video fixed-fusion experiments and external-
cohort outputs that are not reported in the manuscript or Supplementary
Information.
Those restricted inputs are not required to inspect the protocol logic,
publication tables, aggregate figure data, or structural consistency checks.
Authorized users can place data obtained under the source dataset terms into
the paths documented by the configuration and data-loading code before
re-running data-dependent experiments.

Suggested checks from the archive root:

    python -m unittest discover -s tests
    python scripts/build_revision6_source.py --check
    python scripts/generate_revision6_publication.py --check
    python scripts/build_anonymous_supplement.py --check

Machine-checked means only that structure, hashes, propagation rules, and
declared invariants are checked. It does not establish substantive validity,
construct validity, clinical utility, or generalizability.

MANIFEST.json records SHA-256 hashes and sizes for every payload member.
MANIFEST.sha256 is a conventional checksum list. The ZIP uses sorted members,
a fixed timestamp, and fixed permissions so identical inputs produce identical
bytes.
"""
    return text.encode("utf-8")


def build_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {
        f"{ARCHIVE_ROOT}/README.md": readme_payload(),
    }
    for path in collect_files():
        relative = path.relative_to(ROOT)
        name = archive_name(relative)
        assert_safe_member(name)
        payloads[name] = path.read_bytes()

    for name, payload in payloads.items():
        assert_safe_member(name)
        assert_anonymous_text(name, payload)
        assert_no_legacy_results(name, payload)

    records = [
        {
            "path": name.removeprefix(f"{ARCHIVE_ROOT}/"),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(payloads.items())
    ]
    manifest = {
        "archive": OUTPUT.name,
        "archive_root": ARCHIVE_ROOT,
        "created_for": "CHI 2027 anonymous review",
        "deterministic_timestamp": "2026-08-14T00:00:00Z",
        "payload_file_count": len(records),
        "files": records,
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_name = f"{ARCHIVE_ROOT}/MANIFEST.json"
    payloads[manifest_name] = manifest_payload

    checksum_lines = [
        f"{sha256_bytes(payload)}  {name.removeprefix(f'{ARCHIVE_ROOT}/')}"
        for name, payload in sorted(payloads.items())
    ]
    payloads[f"{ARCHIVE_ROOT}/MANIFEST.sha256"] = (
        "\n".join(checksum_lines) + "\n"
    ).encode()
    return payloads


def zip_bytes(payloads: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name])
    return buffer.getvalue()


def validate_archive(payload: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("Archive members must be unique and sorted")
        for name in names:
            assert_safe_member(name)
            member_payload = archive.read(name)
            assert_anonymous_text(name, member_payload)
            assert_no_legacy_results(name, member_payload)
        required = {
            f"{ARCHIVE_ROOT}/README.md",
            f"{ARCHIVE_ROOT}/MANIFEST.json",
            f"{ARCHIVE_ROOT}/MANIFEST.sha256",
            f"{ARCHIVE_ROOT}/supplementary_information.pdf",
            f"{ARCHIVE_ROOT}/results/revision6_source.json",
        }
        missing = required.difference(names)
        if missing:
            raise ValueError(f"Required archive members missing: {sorted(missing)}")

        manifest = json.loads(archive.read(f"{ARCHIVE_ROOT}/MANIFEST.json"))
        indexed = {
            f"{ARCHIVE_ROOT}/{record['path']}": record
            for record in manifest["files"]
        }
        expected = set(names) - {
            f"{ARCHIVE_ROOT}/MANIFEST.json",
            f"{ARCHIVE_ROOT}/MANIFEST.sha256",
        }
        if set(indexed) != expected:
            raise ValueError("Manifest membership does not match archive payload")
        for name in expected:
            body = archive.read(name)
            if indexed[name]["sha256"] != sha256_bytes(body):
                raise ValueError(f"Manifest hash mismatch: {name}")
            if indexed[name]["size_bytes"] != len(body):
                raise ValueError(f"Manifest size mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless the committed archive exactly matches current inputs.",
    )
    args = parser.parse_args()

    payload = zip_bytes(build_payloads())
    validate_archive(payload)
    if args.check:
        if not OUTPUT.exists():
            print(f"Missing archive: {OUTPUT.relative_to(ROOT)}", file=sys.stderr)
            return 1
        if OUTPUT.read_bytes() != payload:
            print("Anonymous supplementary archive is stale", file=sys.stderr)
            return 1
        print(
            f"Anonymous supplementary archive is current: "
            f"{len(payload)} bytes, sha256={sha256_bytes(payload)}"
        )
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, OUTPUT)
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)}: "
        f"{len(payload)} bytes, sha256={sha256_bytes(payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
