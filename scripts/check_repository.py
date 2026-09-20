#!/usr/bin/env python3
"""Read-only pre-publication audit of tracked and non-ignored Git candidates.

This is a conservative local safety check, not a complete secret scanner or a
license/anonymity review. It never stages files, reads credential-named files,
contacts the network, or prints matched secret values. Missing tracked files
are warnings because an unstaged move/delete is normal during organization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
WARNING_BYTES = 10 * MIB
MAXIMUM_BYTES = 100 * MIB
TEXT_SCAN_BYTES = 2 * MIB
PRIVATE_ROOTS = frozenset({
    ".git", ".local", ".venv", "archive", "artifacts", "checkpoints", "references",
    "paper", "paper_zh", "build", "dist",
})
PRIVATE_PARTS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
})
PUBLIC_DATA_FILES = frozenset({"data/DATASET.md", "data/stimuli/PROVENANCE.md"})
# Match the repository's global .gitignore rules even if git add -f was used.
# These formats can carry restricted inputs, opaque model weights or bundles;
# reject by filename before reading content, regardless of size or directory.
RAW_INPUT_SUFFIXES = (
    ".npy", ".npz", ".mat", ".edf", ".set", ".snirf",
    ".mp4", ".mkv", ".avi", ".mov", ".wav",
)
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")
CREDENTIAL_NAME = re.compile(
    r"(?:^|[_-])(?:tokens?|credentials?|secrets?)(?:[_-].*)?"
    r"\.(?:json|ya?ml|toml|ini|txt|csv)$", re.IGNORECASE
)
SECRET_PATTERNS = (
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b")),
    ("AWS access-key ID", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private-key header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+[^)]*)?\)")


@dataclass(frozen=True)
class Issue:
    severity: str
    path: str
    message: str
    line: int | None = None

    def format(self) -> str:
        # Avoid control characters in Git filenames changing terminal output.
        safe_path = self.path.encode("unicode_escape").decode("ascii") if any(
            ord(char) < 32 for char in self.path
        ) else self.path
        location = f"{safe_path}:{self.line}" if self.line else safe_path
        return f"{self.severity.upper()} {location}: {self.message}"


def git_candidates(root: Path) -> list[str]:
    """Include tracked-but-ignored files; never mutate the index."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others",
         "--exclude-standard", "-z"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError("Cannot enumerate Git candidates; run inside a Git repository.")
    return sorted({value.decode("utf-8", errors="surrogateescape")
                   for value in result.stdout.split(b"\0") if value})


def private_path_reason(relative: str) -> str | None:
    path = PurePosixPath(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return "invalid repository-relative path"
    if path.parts[0] in PRIVATE_ROOTS:
        return "local-only directory must not be published"
    if any(part in PRIVATE_PARTS for part in path.parts):
        return "environment/cache directory must not be published"
    if path.parts[0] == "data" and relative not in PUBLIC_DATA_FILES:
        return "raw/input data is private; only the two data documentation files are allowed"
    name = path.name.lower()
    if name.endswith(RAW_INPUT_SUFFIXES):
        return "raw array/recording/media file must not be published (contents not inspected)"
    if name.endswith(WEIGHT_SUFFIXES):
        return "model-weight/checkpoint file must not be published (contents not inspected)"
    if name.endswith(ARCHIVE_SUFFIXES):
        return "opaque archive/bundle file must not be published (contents not inspected)"
    if name == ".env" or (name.startswith(".env.") and name not in {
        ".env.example",
    }):
        return "environment credential/configuration file must not be published"
    if CREDENTIAL_NAME.search(name) or name.endswith((".pem", ".key", ".token", ".p12", ".pfx")):
        return "credential-named file must not be published (contents not inspected)"
    if name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}:
        return "SSH private-key filename must not be published (contents not inspected)"
    if name.endswith((".pyc", ".pyo", ".orig", ".bak", ".swp", ".swo")):
        return "cache/backup/editor file must not be published"
    return None


def secret_issues(relative: str, text: str) -> list[Issue]:
    issues = []
    for number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                issues.append(Issue("error", relative,
                                    f"possible {label}; matched value suppressed", number))
    return issues


def readme_link_issues(
    root: Path, relative: str, text: str, published_paths: set[str] | None = None,
) -> list[Issue]:
    """Check inline local Markdown links, ignoring code fences."""
    issues = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        for match in MARKDOWN_LINK.finditer(line):
            target = match.group(1).strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = unquote(parsed.path)
            # GitHub treats /path as repository-root-relative, not a host path.
            resolved = (root / destination.lstrip("/")).resolve() if destination.startswith("/") else (
                root / relative
            ).parent.joinpath(destination).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                issues.append(Issue("error", relative, "local Markdown link leaves repository", number))
                continue
            if not resolved.exists():
                issues.append(Issue("error", relative, "local Markdown link target is missing", number))
            elif published_paths is not None:
                target_path = resolved.relative_to(root.resolve()).as_posix()
                included = target_path == "." or target_path in published_paths or any(
                    item.startswith(target_path + "/") for item in published_paths
                )
                if not included:
                    issues.append(Issue("error", relative, "local Markdown link target is excluded from publication", number))
    return issues


def audit_repository(root: Path, candidates: list[str] | None = None) -> tuple[list[str], list[Issue]]:
    candidates = git_candidates(root) if candidates is None else sorted(set(candidates))
    published_paths = {
        relative for relative in candidates
        if private_path_reason(relative) is None and (root / relative).is_file()
    }
    issues: list[Issue] = []
    for relative in candidates:
        path = root / relative
        if path.is_symlink():
            issues.append(Issue("error", relative, "symbolic link is not inspected; review/remove it before publication"))
            continue
        if not path.exists():
            issues.append(Issue("warning", relative, "tracked/candidate path is absent; review the pending move/deletion"))
            continue
        reason = private_path_reason(relative)
        if reason:
            issues.append(Issue("error", relative, reason))
            continue
        if not path.is_file():
            issues.append(Issue("error", relative, "candidate is not a regular file"))
            continue
        try:
            size = path.stat().st_size
            if size >= MAXIMUM_BYTES:
                issues.append(Issue("error", relative, "file is at least 100 MiB; exclude it or explicitly plan Git LFS"))
            elif size >= WARNING_BYTES:
                issues.append(Issue("warning", relative, "file is at least 10 MiB; review whether it belongs in Git"))
            if size > TEXT_SCAN_BYTES:
                issues.append(Issue("warning", relative, "content exceeds 2 MiB text-scan limit; review contents separately"))
                continue
            raw = path.read_bytes()
        except OSError:
            issues.append(Issue("error", relative, "cannot read candidate for safety audit"))
            continue
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        issues.extend(secret_issues(relative, text))
        if path.suffix.lower() == ".md":
            issues.extend(readme_link_issues(root, relative, text, published_paths))
    return candidates, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this script's parent repository)")
    args = parser.parse_args(argv)
    try:
        candidates, issues = audit_repository(args.root.resolve())
    except (RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    for issue in issues:
        print(issue.format())
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    print(f"Checked {len(candidates)} Git candidate paths: {errors} error(s), {warnings} warning(s).")
    print("Read-only candidate audit only: no history, binary-content, license, or anonymity clearance; no files staged.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
