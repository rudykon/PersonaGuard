from __future__ import annotations

from collections import Counter
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def latex_values(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text, flags=re.DOTALL)


def citation_keys(text: str) -> Counter[str]:
    return Counter(
        key.strip()
        for group in latex_values(r"\\cite\{([^}]+)\}", text)
        for key in group.split(",")
    )


@unittest.skipUnless(
    (PAPER / "main_zh.tex").is_file(),
    "Chinese manuscript is not part of the current publication",
)
class ChineseLatexTest(unittest.TestCase):
    def test_every_english_tex_has_a_chinese_counterpart(self) -> None:
        english = sorted(
            path
            for path in PAPER.rglob("*.tex")
            if not path.stem.endswith("_zh")
        )
        self.assertTrue(english)
        for source in english:
            translated = source.with_name(source.stem + "_zh.tex")
            self.assertTrue(translated.is_file(), translated)
            text = translated.read_text(encoding="utf-8")
            self.assertRegex(text, r"[\u3400-\u9fff]", translated)
            controls = [
                (index, ord(char))
                for index, char in enumerate(text)
                if ord(char) < 32 and char != "\n"
            ]
            self.assertEqual(controls, [], translated)

    def test_main_and_supplement_use_chinese_dependencies(self) -> None:
        for english_name, chinese_name in (
            ("main.tex", "main_zh.tex"),
            ("body.tex", "body_zh.tex"),
            ("supplementary_information.tex", "supplementary_information_zh.tex"),
        ):
            english = (PAPER / english_name).read_text(encoding="utf-8")
            chinese = (PAPER / chinese_name).read_text(encoding="utf-8")
            for dependency in latex_values(r"\\input\{([^}]+)\}", english):
                self.assertIn(rf"\input{{{dependency}_zh}}", chinese)

        for entrypoint in ("main_zh.tex", "supplementary_information_zh.tex"):
            text = (PAPER / entrypoint).read_text(encoding="utf-8")
            self.assertIn("pdflang={zh-CN}", text)
            self.assertIn("pdfauthor={匿名作者}", text)

    def test_body_and_supplement_preserve_source_anchors(self) -> None:
        for english_name, chinese_name in (
            ("body.tex", "body_zh.tex"),
            ("supplementary_information.tex", "supplementary_information_zh.tex"),
        ):
            english = (PAPER / english_name).read_text(encoding="utf-8")
            chinese = (PAPER / chinese_name).read_text(encoding="utf-8")
            self.assertEqual(
                set(latex_values(r"\\label\{([^}]+)\}", english)),
                set(latex_values(r"\\label\{([^}]+)\}", chinese)),
            )
            self.assertEqual(citation_keys(english), citation_keys(chinese))
            self.assertEqual(
                Counter(
                    latex_values(
                        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                        english,
                    )
                ),
                Counter(
                    latex_values(
                        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                        chinese,
                    )
                ),
            )
            self.assertEqual(
                Counter(latex_values(r"\\begin\{([^}]+)\}", english)),
                Counter(latex_values(r"\\begin\{([^}]+)\}", chinese)),
            )
            for command in ("section", "subsection", "paragraph"):
                self.assertEqual(
                    len(latex_values(rf"\\{command}\{{", english)),
                    len(latex_values(rf"\\{command}\{{", chinese)),
                )

    def test_body_has_exact_latex_command_parity(self) -> None:
        command = re.compile(r"\\[A-Za-z@]+|\\[%&{}_#$]|\\\\")
        english = (PAPER / "body.tex").read_text(encoding="utf-8")
        chinese = (PAPER / "body_zh.tex").read_text(encoding="utf-8")
        self.assertEqual(Counter(command.findall(english)), Counter(command.findall(chinese)))
        self.assertEqual(english.count(r"\("), chinese.count(r"\("))
        self.assertEqual(english.count(r"\)"), chinese.count(r"\)"))
        level = 0
        for token in re.findall(r"\\\(|\\\)", chinese):
            level += 1 if token == r"\(" else -1
            self.assertIn(level, (0, 1), "nested or unmatched inline math")
        self.assertEqual(level, 0, "unclosed inline math")

    def test_no_translation_placeholders_or_residual_status_prose(self) -> None:
        joined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(PAPER.rglob("*_zh.tex"))
        )
        for token in ("TODO", "TRANSLATE", "PLACEHOLDER", "待翻译", "Track4"):
            self.assertNotIn(token, joined)
        evidence = (
            PAPER / "generated" / "evidence_traceability_rows_zh.tex"
        ).read_text(encoding="utf-8")
        for token in (
            "PARTIAL",
            "NOT TESTED",
            "NOT JUSTIFIED",
            "participant-held-out",
            "video-only correction",
            " donor ",
            " joystick",
        ):
            self.assertNotIn(token, evidence)
        ethics = (
            PAPER / "generated" / "secondary_analysis_ethics_statement_zh.tex"
        ).read_text(encoding="utf-8")
        self.assertNotIn("gated research release", ethics)


if __name__ == "__main__":
    unittest.main()
