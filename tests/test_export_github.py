"""Verify the upload archive's boundaries in disposable Git repositories."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from scripts.export_github import export_repository


class GitHubExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="chi2027-export-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.put(".gitignore", "/dist/\n/paper/\n/paper_zh/\n/data/*\n!/data/DATASET.md\n/.local/\n")
        self.put("README.md", "[Data](data/DATASET.md)\n")
        self.put("data/DATASET.md", "Data access instructions.\n")

    def put(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_exports_current_uncommitted_files_without_local_material_or_history(self):
        self.put("src/current.py", "old = True\n")
        subprocess.run(["git", "-C", str(self.root), "add", "src/current.py"], check=True)
        self.put("src/current.py", "current = True\n")
        self.put("src/new.py", "new = True\n")
        self.put("paper/main.tex", "local manuscript")
        self.put("paper_zh/main.tex", "local translation")
        self.put("data/raw.txt", "private data")
        self.put(".local/token.json", "private fixture")
        # Even old tracked manuscript files must not enter a fresh source export.
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "paper/main.tex"], check=True)
        index_before = (self.root / ".git/index").read_bytes()
        output, count = export_repository(self.root)
        with zipfile.ZipFile(output) as bundle:
            self.assertEqual(set(bundle.namelist()), {
                "CHI2027/.gitignore", "CHI2027/README.md", "CHI2027/data/DATASET.md",
                "CHI2027/src/current.py", "CHI2027/src/new.py",
            })
            self.assertEqual(count, 5)
            self.assertEqual(bundle.read("CHI2027/src/current.py"), b"current = True\n")
        self.assertEqual((self.root / ".git/index").read_bytes(), index_before)
        self.assertTrue((self.root / "paper/main.tex").is_file())

    def test_pending_deletions_are_omitted(self):
        path = self.put("old.py", "old = True\n")
        subprocess.run(["git", "-C", str(self.root), "add", "old.py"], check=True)
        path.unlink()
        output, _ = export_repository(self.root)
        with zipfile.ZipFile(output) as bundle:
            self.assertNotIn("CHI2027/old.py", bundle.namelist())

    def test_failed_secret_audit_preserves_existing_archive_and_hides_value(self):
        output, _ = export_repository(self.root)
        original = output.read_bytes()
        secret = "gh" + "p_" + "A" * 36
        self.put("accidental.py", "token = '" + secret + "'\n")
        log = io.StringIO()
        with redirect_stdout(log), self.assertRaises(RuntimeError):
            export_repository(self.root)
        self.assertNotIn(secret, log.getvalue())
        self.assertEqual(output.read_bytes(), original)

    def test_link_to_excluded_manuscript_blocks_export(self):
        self.put("paper/main.tex", "local manuscript")
        self.put("docs/guide.md", "[Manuscript](../paper/main.tex)\n")
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            export_repository(self.root)
        self.assertFalse((self.root / "dist").exists())

    def test_source_symlink_is_rejected(self):
        (self.root / "linked.py").symlink_to(self.root / "README.md")
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            export_repository(self.root)

    def test_repeated_export_replaces_generated_zip_without_including_it(self):
        output, count = export_repository(self.root)
        self.put("src/new.py", "new = True\n")
        updated, new_count = export_repository(self.root)
        self.assertEqual(output, updated)
        self.assertEqual(new_count, count + 1)
        with zipfile.ZipFile(output) as bundle:
            self.assertIn("CHI2027/src/new.py", bundle.namelist())
            self.assertFalse(any("/dist/" in name for name in bundle.namelist()))


if __name__ == "__main__":
    unittest.main()
