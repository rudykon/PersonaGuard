"""Self-contained tests for the read-only Git publication safety check."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_repository as check


class RepositorySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chi2027-repository-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def put(self, relative, content="ordinary content\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def audit(self, *candidates):
        return check.audit_repository(self.root, list(candidates))[1]

    def test_private_directories_and_raw_data_are_rejected(self):
        for relative in (".local/notes.txt", "archive/old.tex", "artifacts/results.json",
                         "checkpoints/model.pt", ".venv/file", "references/article.pdf",
                         "data/stimuli/stimulus_audit.json", "data/raw.json"):
            with self.subTest(relative=relative):
                self.put(relative)
                self.assertEqual(self.audit(relative)[0].severity, "error")

    def test_public_source_data_and_data_documentation_are_allowed(self):
        for relative in ("paper_support/revision6_source.json", "paper_support/figures/source.csv",
                         "paper/main.pdf", "paper/figures/protocol_embed.pdf",
                         "data/DATASET.md", "data/stimuli/PROVENANCE.md", ".env.example"):
            self.put(relative)
            self.assertEqual(self.audit(relative), [])

    def test_small_raw_weights_and_archives_rejected_without_reading(self):
        suffixes = check.RAW_INPUT_SUFFIXES + check.WEIGHT_SUFFIXES + check.ARCHIVE_SUFFIXES
        for suffix in suffixes:
            relative = "otherwise_public/tiny" + suffix
            with self.subTest(relative=relative):
                self.put(relative, "small synthetic fixture")
                with patch.object(Path, "read_bytes", side_effect=AssertionError("must not inspect private binary")):
                    issue = self.audit(relative)[0]
                self.assertEqual(issue.severity, "error")
                self.assertIn("contents not inspected", issue.message)

    def test_private_suffix_checks_are_case_insensitive(self):
        self.put("paper_support/ACCIDENTAL.NPZ", "synthetic fixture")
        self.assertEqual(self.audit("paper_support/ACCIDENTAL.NPZ")[0].severity, "error")

    def test_credential_names_rejected_without_reading_contents(self):
        for relative in ("github_token.json", "token.json", "credentials.yaml", ".env",
                         ".env.production", ".env.sample", ".env.template", "key.pem", "id_ed25519"):
            self.put(relative)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must not read credentials")):
                self.assertEqual(self.audit(relative)[0].severity, "error")

    def test_matched_values_never_appear_in_issues(self):
        token = "gh" + "p_" + "A" * 36
        self.put("example.py", "# public line\nvalue = '" + token + "'\n")
        issues = self.audit("example.py")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].line, 2)
        self.assertNotIn(token, issues[0].format())

    def test_plain_hashes_and_scientific_numbers_are_not_secrets(self):
        self.put("numbers.json", '{"sha256": "' + "a" * 64 + '", "score": 0.123456789}')
        self.assertEqual(self.audit("numbers.json"), [])

    def test_missing_tracked_path_is_only_warning(self):
        issue = self.audit("paper/figures/moved.svg")[0]
        self.assertEqual(issue.severity, "warning")

    def test_symlinks_are_not_followed(self):
        (self.root / "linked.json").symlink_to(self.root / "missing-secret.json")
        self.assertIn("symbolic link", self.audit("linked.json")[0].message)

    def test_large_file_limit(self):
        self.put("big.txt", "size fixture")
        with patch.object(check, "MAXIMUM_BYTES", 8), patch.object(check, "WARNING_BYTES", 4):
            self.assertEqual(self.audit("big.txt")[0].severity, "error")

    def test_readme_links_are_checked_and_code_fences_skipped(self):
        self.put("docs/README.md")
        self.put("README.md", "[docs](docs/README.md#section)\n[web](https://example.org)\n"
                 "```\n[example](missing-in-code.md)\n```\n[missing](missing.md)\n")
        issues = self.audit("README.md")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].line, 6)

    def test_git_enumeration_keeps_tracked_ignored_paths(self):
        self.put(".gitignore", "tracked_token.json\nignored_token.json\n")
        self.put("tracked_token.json")
        self.put("ignored_token.json")
        self.put("source.py")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        # This is an isolated disposable fixture, never the real repository.
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "tracked_token.json"], check=True)
        candidates, issues = check.audit_repository(self.root)
        self.assertIn("tracked_token.json", candidates)
        self.assertNotIn("ignored_token.json", candidates)
        self.assertTrue(any(issue.path == "tracked_token.json" and issue.severity == "error" for issue in issues))

    def test_forced_tracked_small_archive_is_still_rejected(self):
        self.put(".gitignore", "*.zip\n")
        self.put("tiny.zip", "synthetic non-sensitive fixture")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        # Force-tracking occurs only in this disposable fixture repository.
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "tiny.zip"], check=True)
        candidates, issues = check.audit_repository(self.root)
        self.assertIn("tiny.zip", candidates)
        self.assertTrue(any(issue.path == "tiny.zip" and issue.severity == "error" for issue in issues))

    def test_main_fails_on_errors_but_not_missing_paths(self):
        self.put("token.json")
        with patch.object(check, "git_candidates", return_value=["token.json"]), redirect_stdout(io.StringIO()):
            self.assertEqual(check.main(["--root", str(self.root)]), 1)
        with patch.object(check, "git_candidates", return_value=["missing.txt"]), redirect_stdout(io.StringIO()):
            self.assertEqual(check.main(["--root", str(self.root)]), 0)


if __name__ == "__main__":
    unittest.main()
