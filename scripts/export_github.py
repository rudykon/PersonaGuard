#!/usr/bin/env python3
"""Export audited working-tree source files without local material or Git history."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

if __package__:
    from . import check_repository as check
else:
    import check_repository as check


def export_repository(root: Path) -> tuple[Path, int]:
    """Write a source ZIP; never stage, commit, or change the original files."""
    root = root.resolve()
    candidates = []
    for relative in check.git_candidates(root):
        path = PurePosixPath(relative)
        if path.parts[0] in check.PRIVATE_ROOTS:
            continue
        if path.parts[0] == "data" and relative not in check.PUBLIC_DATA_FILES:
            continue
        source = root / relative
        # Pending deletions are not part of a working-tree export.
        if source.exists() or source.is_symlink():
            candidates.append(relative)

    if not candidates:
        raise RuntimeError("No public source files found.")
    _, issues = check.audit_repository(root, candidates)
    for issue in issues:
        print(issue.format())
    if any(issue.severity == "error" for issue in issues):
        raise RuntimeError("Export stopped: fix the candidate audit errors first.")

    output_dir = root / "dist"
    if output_dir.is_symlink():
        raise RuntimeError("Export directory must not be a symbolic link.")
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "CHI2027-github.zip"
    # Build separately so a failed export cannot destroy an existing archive.
    with tempfile.TemporaryDirectory(prefix=".github-export-", dir=output_dir) as temporary:
        archive = Path(temporary) / output.name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative in candidates:
                bundle.write(root / relative, arcname=f"CHI2027/{relative}")
        archive.replace(output)
    return output, len(candidates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=check.ROOT, help="source Git repository")
    args = parser.parse_args(argv)
    try:
        output, count = export_repository(args.root)
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Exported {count} files ({output.stat().st_size / check.MIB:.2f} MiB): {output}")
    print("Working-tree snapshot only; no Git history, staging, commit, or upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
