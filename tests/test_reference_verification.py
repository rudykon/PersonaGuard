import unittest

from scripts import verify_references as verifier


class ReferenceVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = verifier.parse_bibtex(
            verifier.BIB_PATH.read_text(encoding="utf-8")
        )

    def test_human_readable_report_can_be_rendered_without_stored_audit(self):
        report = {
            "audit_date": "2026-01-01",
            "entry_count": 1,
            "summary": {"verified": 1, "needs_fix": 0, "unverifiable": 0},
            "entries": [{
                "key": "example", "status": "verified", "confirmed_sources": 2,
                "best_title_similarity": 1.0, "issues": [], "sources": [],
                "fields": {"author": "Doe, Jane", "title": "Example Article", "year": "2026"},
            }],
        }
        text = verifier.markdown_report(report)
        self.assertIn("Entries parsed: 1", text)
        self.assertIn("needs_fix=0", text)
        self.assertIn("unverifiable=0", text)
        self.assertIn("### example", text)
        self.assertTrue(verifier.JSON_OUTPUT.is_relative_to(verifier.ROOT / "artifacts"))
        self.assertTrue(verifier.MARKDOWN_OUTPUT.is_relative_to(verifier.ROOT / "artifacts"))

    def test_frontiers_article_number_is_recovered_from_doi(self):
        self.assertEqual(
            verifier.doi_article_number("10.3389/fnhum.2012.00112"),
            "112",
        )
        self.assertEqual(
            verifier.doi_article_number("https://doi.org/10.3389/fninf.2010.00005"),
            "5",
        )
        self.assertEqual(verifier.doi_article_number("10.1145/123.456"), "")

    def test_preferred_article_number_beats_lower_priority_page_noise(self):
        entry = {
            "key": "frontiers-example",
            "type": "article",
            "fields": {
                "author": "Doe, Jane",
                "title": "Example Article",
                "year": "2012",
                "volume": "6",
                "pages": "112",
                "doi": "10.3389/fnhum.2012.00112",
            },
        }
        crossref = {
            "source": "Crossref",
            "status": "ok",
            "title_similarity": 1.0,
            "metadata": {
                "title": "Example Article",
                "authors": ["Jane Doe"],
                "year": "2012",
                "volume": "6",
                "pages": "",
                "article_number": "112",
                "doi": "10.3389/fnhum.2012.00112",
            },
        }
        official = {
            "source": "Official page",
            "status": "ok",
            "title_similarity": 1.0,
            "metadata": {
                "title": "Example Article",
                "authors": ["Jane Doe"],
                "year": "2012",
                "volume": "6",
                "pages": "23556",
                "article_number": "",
                "doi": "10.3389/fnhum.2012.00112",
            },
        }

        result = verifier.compare_entry(entry, [crossref, official])

        self.assertEqual(result["status"], "verified")
        self.assertFalse(
            any(issue["field"] == "pages" for issue in result["issues"])
        )

    def test_article_number_is_not_compared_with_a_page_range(self):
        entry = {
            "key": "acm-example",
            "type": "inproceedings",
            "fields": {
                "author": "Doe, Jane",
                "title": "Example ACM Article",
                "year": "2019",
                "articleno": "3",
                "numpages": "13",
                "doi": "10.1145/3290605.3300233",
            },
        }
        source = {
            "source": "Crossref",
            "status": "ok",
            "title_similarity": 1.0,
            "metadata": {
                "title": "Example ACM Article",
                "authors": ["Jane Doe"],
                "year": "2019",
                "pages": "1-13",
                "article_number": "",
                "doi": "10.1145/3290605.3300233",
            },
        }

        result = verifier.compare_entry(entry, [source])

        self.assertNotEqual(result["status"], "needs_fix")
        self.assertFalse(
            any(issue["field"] == "article_number" for issue in result["issues"])
        )


    def test_middle_author_swap_is_detected(self):
        entry = {"key": "author-order", "type": "article", "fields": {
            "author": "Doe, Jane and Brown, Alex and Smith, Robin",
            "title": "Example Article", "year": "2026"}}
        source = {"source": "Crossref", "status": "ok", "title_similarity": 1.0,
                  "metadata": {"title": "Example Article", "year": "2026",
                               "authors": ["Jane Doe", "Robin Smith", "Alex Brown"]}}
        result = verifier.compare_entry(entry, [source])
        self.assertEqual(result["status"], "needs_fix")
        self.assertTrue(any(issue["field"] == "author_order" for issue in result["issues"]))

    def test_mer_reference_is_pinned_to_the_cited_author_version(self):
        fields = next(e["fields"] for e in self.entries if e["key"] == "lian2026mer")
        self.assertEqual(fields["eprint"], "2604.19417v4")
        self.assertEqual(fields["url"], "https://arxiv.org/abs/2604.19417v4")
        self.assertEqual(len(verifier.split_bib_authors(fields["author"])), 18)


    def test_crossref_separate_subtitle_is_retained(self):
        metadata = verifier.crossref_metadata({"title": ["Main title"], "subtitle": ["Full subtitle"]})
        self.assertEqual(metadata["title"], "Main title: Full subtitle")


if __name__ == "__main__":
    unittest.main()
