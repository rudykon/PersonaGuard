import io
import json
import unittest
import zipfile

from scripts import build_anonymous_supplement as builder


class AnonymousSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not builder.OUTPUT.exists():
            raise AssertionError(
                "Build paper/submission/chi2027_anonymous_supplement.zip first"
            )
        cls.payload = builder.OUTPUT.read_bytes()

    def test_archive_matches_deterministic_builder(self):
        expected = builder.zip_bytes(builder.build_payloads())
        self.assertEqual(self.payload, expected)

    def test_archive_validation_passes(self):
        builder.validate_archive(self.payload)

    def test_members_are_sorted_and_exclude_restricted_inputs(self):
        with zipfile.ZipFile(io.BytesIO(self.payload)) as archive:
            names = archive.namelist()
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            builder.assert_safe_member(name)
            self.assertFalse(
                name.endswith("_zh.tex"), msg=f"Chinese reading copy included: {name}"
            )

    def test_manifest_covers_every_payload_member(self):
        with zipfile.ZipFile(io.BytesIO(self.payload)) as archive:
            names = set(archive.namelist())
            manifest_name = f"{builder.ARCHIVE_ROOT}/MANIFEST.json"
            checksum_name = f"{builder.ARCHIVE_ROOT}/MANIFEST.sha256"
            manifest = json.loads(archive.read(manifest_name))
            indexed = {
                f"{builder.ARCHIVE_ROOT}/{record['path']}": record
                for record in manifest["files"]
            }
            expected = names - {manifest_name, checksum_name}
            self.assertEqual(set(indexed), expected)
            for name in expected:
                payload = archive.read(name)
                self.assertEqual(
                    indexed[name]["sha256"],
                    builder.sha256_bytes(payload),
                )
                self.assertEqual(indexed[name]["size_bytes"], len(payload))

    def test_text_payloads_do_not_leak_private_paths(self):
        with zipfile.ZipFile(io.BytesIO(self.payload)) as archive:
            for name in archive.namelist():
                builder.assert_anonymous_text(name, archive.read(name))

    def test_superseded_known_video_materials_are_excluded(self):
        with zipfile.ZipFile(io.BytesIO(self.payload)) as archive:
            names = archive.namelist()
            for name in names:
                relative = name.removeprefix(f"{builder.ARCHIVE_ROOT}/")
                self.assertFalse(
                    builder.is_legacy_known_video_path(relative),
                    msg=f"Superseded known-video member included: {name}",
                )
                builder.assert_no_legacy_results(name, archive.read(name))

    def test_current_chi_algorithm_routes_remain_packaged(self):
        required = {
            f"{builder.ARCHIVE_ROOT}/scripts/run_content_prior_experiments.py",
            f"{builder.ARCHIVE_ROOT}/scripts/run_causal_physio_residual.py",
            f"{builder.ARCHIVE_ROOT}/scripts/run_sparse_anchor_optimization.py",
            f"{builder.ARCHIVE_ROOT}/scripts/run_protocol_replay.py",
            f"{builder.ARCHIVE_ROOT}/src/merps/content_prior.py",
            f"{builder.ARCHIVE_ROOT}/src/merps/physiology_residual.py",
            f"{builder.ARCHIVE_ROOT}/src/merps/sparse_anchor.py",
        }
        with zipfile.ZipFile(io.BytesIO(self.payload)) as archive:
            self.assertTrue(required.issubset(archive.namelist()))

    def test_portable_publication_source_is_packaged(self):
        member = f"{builder.ARCHIVE_ROOT}/results/revision6_source.json"
        with zipfile.ZipFile(io.BytesIO(self.payload)) as archive:
            source = archive.read(member).decode("utf-8")
        self.assertNotIn("/home/", source)
        self.assertNotIn("/Users/", source)
        self.assertNotIn("file" + "://", source)


if __name__ == "__main__":
    unittest.main()
