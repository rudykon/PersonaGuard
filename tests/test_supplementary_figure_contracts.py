"""Guard four supplementary figures against evidence and numbering drift."""
import csv
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"paper/figures"
SCRIPTS=ROOT/"scripts/figures"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("si_figures",SCRIPTS/"make_supplementary_figures.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def rows(stem):
    with (SCRIPTS/f"source_data_{stem}.csv").open() as f:
        return list(csv.DictReader(f))

class SupplementaryFigureContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=json.loads((ROOT/"results/revision6_source.json").read_text())
        cls.s=cls.source["summaries"]

    def test_all_graph_nodes_states_and_directions_preserved(self):
        data=rows("measurement_evidence_dag_full")
        nodes=[r for r in data if r["kind"]=="node"]
        edges=[r for r in data if r["kind"]=="edge"]
        graph=self.s["evidence_traceability"]
        self.assertEqual(len(nodes),11); self.assertEqual(len(edges),14)
        self.assertEqual({r["node"]:r["status"] for r in nodes},
                         {r["id"]:r["status"] for r in graph["nodes"]})
        self.assertEqual({(r["source"],r["target"],r["type"]) for r in edges},
                         {(r["source"],r["target"],r["type"]) for r in graph["edges"]})
        self.assertEqual(set(module.ROUTES),{(e["source"],e["target"]) for e in graph["edges"]})

    def test_walkthrough_is_existing_case_not_new_node(self):
        data=[r for r in rows("measurement_evidence_dag_full") if r["panel"]=="B"]
        self.assertEqual(len(data),1)
        case=next(r for r in self.s["protocol_replay_cases"]["cases"] if r["id"]=="signed_calibration")
        self.assertEqual(data[0]["source_locator"],case["source_locator"])
        for key in ["comparator_increment","evaluation_mode","user_outcome"]:
            self.assertEqual(data[0][key],case["evidence"][key])
        self.assertEqual(data[0]["action"],"RUN_PREREGISTERED_USER_STUDY")

    def test_information_boundaries_use_manifest_folds(self):
        data=rows("validation_information_boundaries")
        for actual,expected in zip(data,module.boundary_records(self.source)):
            for key,value in expected.items():
                self.assertEqual(actual[key],str(value))
        self.assertEqual(len(data),6)
        sam=next(r for r in data if r["condition"]=="known_video_sam")
        self.assertEqual(sam.get("outer_video_folds",""),"")
        self.assertIn("same-video",sam["prediction_input"])
        self.assertIn("no target-video joystick",data[0]["prediction_input"])

    def test_all_algorithm_estimates_match_canonical_source(self):
        actual=rows("dense_algorithm_sensitivity")
        expected=module.algorithm_rows(self.source)
        self.assertEqual(len(actual),20)
        for a,e in zip(actual,expected):
            for k,v in e.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):
                    self.assertEqual(float(a[k]),v)
                else: self.assertEqual(a[k],str(v))

    def test_range_and_interval_meanings_remain_distinct(self):
        data=rows("dense_algorithm_sensitivity")
        a=[r for r in data if r["panel"]=="A"]
        self.assertEqual(len(a),7)
        self.assertTrue(all(r["interval"]=="fixed-offset minimum--maximum; not CI" for r in a))
        self.assertEqual(sum(r["retained"]=="True" for r in a),1)
        c=[r for r in data if r["panel"]=="C"]
        self.assertEqual(sum(bool(r["low"]) for r in c),1)
        self.assertIn("crossed-bootstrap",c[-1]["interval"])

    def test_both_gate_families_preserve_all_evaluated_thresholds(self):
        data=[r for r in rows("dense_algorithm_sensitivity") if r["panel"]=="B"]
        for family in ("full_modalities","temporal_token_modalities"):
            r=[v for v in data if v["condition"]==family]
            self.assertEqual([float(v["threshold"]) for v in r],[0,.05,.10,.20,.50])
            self.assertEqual([float(v["threshold"]) for v in r if v["primary"]=="True"],[.10])
            self.assertTrue(all(v["comparator"]=="frozen content prior" for v in r))

    def test_threshold_counts_and_allocation_intervals_unchanged(self):
        data=rows("decision_boundary_sensitivity")
        expected=self.source["tables"]["decision_threshold_curves"]
        counts=[r for r in data if r["panel"]!="D"]
        self.assertEqual(len(counts),len(expected))
        for a,e in zip(counts,expected):
            for k in ("threshold","count","participants","proportion"):
                self.assertEqual(float(a[k]),float(e[k]))
        allocations=[r for r in data if r["panel"]=="D"]
        exp=[r for r in self.source["tables"]["repeated_grouped_cv"] if r["method"]=="context_plus_blockwise_residual"]
        self.assertEqual(len(allocations),5)
        for a,e in zip(allocations,exp):
            for k in ("repeat","gain_vs_video_mean_seconds","gain_vs_video_mean_ci95_low","gain_vs_video_mean_ci95_high"):
                self.assertEqual(float(a[k]),float(e[k]))
        self.assertFalse(any("wilson" in k.lower() for k in data[0]))

    def test_four_si_figures_in_order_in_both_languages(self):
        labels=["fig:si-full-graph","fig:si-information-boundaries","fig:si-algorithm-sensitivity","fig:si-threshold-curves"]
        for name in ["supplementary_information.tex","supplementary_information_zh.tex"]:
            text=(ROOT/"paper"/name).read_text()
            self.assertEqual(text.count("\\begin{figure"),4)
            positions=[text.index("\\label{"+label+"}") for label in labels]
            self.assertEqual(positions,sorted(positions))
            self.assertIn("\\ref{fig:si-threshold-curves}",text)
            self.assertNotIn("Supplementary Figure~S2",text)
            self.assertNotIn("补充材料图~S2",text)
        self.assertIn("Supplementary Figure~S4",(ROOT/"paper/body.tex").read_text())
        self.assertIn("补充图 S4",(ROOT/"paper/body_zh.tex").read_text())

    def test_manuscript_exports_and_build_registration(self):
        build=(ROOT/"scripts/build_revision6_publication.py").read_text()
        author_stems={"Evidence_Graph","Validation_Boundaries_portraits"}
        for stem in ["Evidence_Graph","Validation_Boundaries_portraits",
                     "dense_algorithm_sensitivity","decision_boundary_sensitivity"]:
            self.assertIn('"'+stem+'"',build)
            # Formal assets keep author originals and cited PDFs only.
            suffixes=[".svg","_embed.pdf"] if stem in author_stems else ["_embed.pdf"]
            for suffix in suffixes:
                self.assertTrue((FIG/(stem+suffix)).is_file())
        script=(SCRIPTS/"make_supplementary_figures.py").read_text()
        self.assertIn('f"{stem}.png"',script)
        self.assertIn('f"{stem}.tiff"',script)
        self.assertIn('f"{stem}_grayscale.png"',script)
        self.assertNotIn(".npz",script)
        self.assertNotIn("fill_between",script)
        self.assertNotIn("twinx",script)

    def test_author_svg_sources_map_to_same_si_figures_in_both_languages(self):
        expected={
            "fig:si-full-graph":"Evidence_Graph_embed.pdf",
            "fig:si-information-boundaries":"Validation_Boundaries_portraits_embed.pdf",
        }
        for manuscript in ("supplementary_information.tex","supplementary_information_zh.tex"):
            text=(ROOT/"paper"/manuscript).read_text()
            blocks=re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",text,re.S)
            for label,filename in expected.items():
                matches=[block for block in blocks if "\\label{"+label+"}" in block]
                self.assertEqual(len(matches),1,(manuscript,label))
                images=re.findall(r"\\includegraphics\s*(?:\[.*?\])?\s*\{([^}]+)\}",matches[0],re.S)
                self.assertEqual(images,[filename],(manuscript,label))
            self.assertNotIn("measurement_evidence_dag_full_embed.pdf",text)
            self.assertNotIn("validation_information_boundaries_embed.pdf",text)

    def test_legacy_si_renderers_export_author_svg_without_writing_legacy_art(self):
        expected={
            "measurement_evidence_dag_full":"Evidence_Graph",
            "validation_information_boundaries":"Validation_Boundaries_portraits",
        }
        for legacy_stem,author_stem in expected.items():
            with self.subTest(stem=legacy_stem):
                figure=Mock()
                with patch.object(module.exporter,"export_one") as export_one, \
                     patch.object(module.plt,"close") as close:
                    module.export(figure,legacy_stem)
                export_one.assert_called_once_with(author_stem)
                close.assert_called_once_with(figure)
                figure.savefig.assert_not_called()

class FigureExportRequirementsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location(
            "publication_export_requirements",
            ROOT/"scripts/build_revision6_publication.py",
        )
        cls.builder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix="chi2027-export-test-")
        self.addCleanup(temporary.cleanup)
        self.directory=Path(temporary.name)
        figures=self.directory/"artifacts/figures"
        figures.mkdir(parents=True)
        for suffix in self.builder.FIGURE_SUFFIXES:
            (figures/("example"+suffix)).write_bytes(b"test fixture")
        published=self.directory/"paper/figures"
        published.mkdir(parents=True)
        (published/"example_embed.pdf").write_bytes(b"test fixture")
        (self.directory/"paper/body.tex").write_text(
            r"\includegraphics{example_embed.pdf}", encoding="utf-8"
        )
        for name,value in [("ROOT",self.directory),("FIGURE_STEMS",("example",)),("CUSTOM_SVG_STEMS",frozenset())]:
            change=patch.object(self.builder,name,value)
            change.start()
            self.addCleanup(change.stop)

    def test_formal_assets_suffice_without_work_exports(self):
        for path in (self.directory/"artifacts/figures").iterdir():
            path.unlink()
        self.builder.verify_figure_exports()

    def test_missing_manuscript_pdf_still_fails(self):
        (self.directory/"paper/figures/example_embed.pdf").unlink()
        with self.assertRaisesRegex(RuntimeError,"example_embed.pdf"):
            self.builder.verify_figure_exports()

    def test_empty_manuscript_pdf_still_fails(self):
        (self.directory/"paper/figures/example_embed.pdf").write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError,"example_embed.pdf"):
            self.builder.verify_figure_exports()

    def test_explicit_raster_check_remains_available(self):
        with self.assertRaisesRegex(RuntimeError,"example.png"):
            self.builder.verify_figure_exports(require_raster=True)
        for suffix in self.builder.RASTER_FIGURE_SUFFIXES:
            (self.directory/"artifacts/figures"/("example"+suffix)).write_bytes(b"test fixture")
        self.builder.verify_figure_exports(require_raster=True)
        self.assertTrue(self.builder.parse_args(["--check","--require-raster"]).require_raster)

    def test_unreferenced_paper_image_is_rejected(self):
        (self.directory/"paper/figures/unused.pdf").write_bytes(b"test fixture")
        with self.assertRaisesRegex(RuntimeError,"Unreferenced"):
            self.builder.verify_figure_exports()

    def test_author_sources_remain_in_paper_and_are_not_overwritten(self):
        self.builder.FIGURE_STEMS=("example","author")
        self.builder.CUSTOM_SVG_STEMS=frozenset({"author"})
        (self.directory/"paper/body.tex").write_text(
            r"\includegraphics{example_embed.pdf}\includegraphics{author_embed.pdf}",
            encoding="utf-8",
        )
        formal=self.directory/"paper/figures"
        (formal/"author.svg").write_bytes(b"original SVG")
        (formal/"author_embed.pdf").write_bytes(b"author export")
        (self.directory/"artifacts/figures/author_embed.pdf").write_bytes(b"unwanted replacement")
        self.builder.publish_figure_exports()
        self.assertEqual((formal/"author.svg").read_bytes(),b"original SVG")
        self.assertEqual((formal/"author_embed.pdf").read_bytes(),b"author export")
        self.builder.verify_figure_exports()
        (formal/"author.svg").unlink()
        with self.assertRaisesRegex(RuntimeError,"author.svg"):
            self.builder.verify_figure_exports()

    def test_publisher_only_copies_registered_cited_images(self):
        figures=self.directory/"artifacts/figures"
        (figures/"unused_embed.pdf").write_bytes(b"unused source")
        (figures/"example_embed.pdf").write_bytes(b"updated fixture")
        self.builder.publish_figure_exports()
        self.assertEqual(
            (self.directory/"paper/figures/example_embed.pdf").read_bytes(),
            b"updated fixture",
        )
        self.assertFalse((self.directory/"paper/figures/unused_embed.pdf").exists())
        self.builder.verify_figure_exports()

if __name__=="__main__":
    unittest.main()
