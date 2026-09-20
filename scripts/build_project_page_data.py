#!/usr/bin/env python3
"""Build the project page's recorded cases from the public protocol resolver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs' / 'assets' / 'cases.js'
COPY = ROOT / 'docs' / 'assets' / 'case-copy.js'


def validate_case_copy(cases: list[dict]) -> None:
    """Fail when explanatory copy drifts from the actual recorded decisions."""
    prefix = 'window.PERSONAGUARD_CASE_COPY = '
    text = COPY.read_text(encoding='utf-8').strip()
    if not text.startswith(prefix) or not text.endswith(';'):
        raise ValueError('Unexpected case-copy.js format')
    copy = json.loads(text[len(prefix):-1])
    if set(copy) != {case['id'] for case in cases}:
        raise ValueError('Presentation case IDs differ from the official case set')
    for case in cases:
        item = copy[case['id']]
        if item['expected_action'] != case['audited_action']:
            raise ValueError(f"Review presentation copy: decision changed for {case['id']}")
        for key in ('question', 'use', 'evidence', 'next', 'why', 'context'):
            for language in ('en', 'zh'):
                if not isinstance(item[key][language], str) or not item[key][language].strip():
                    raise ValueError(f"Missing {language} presentation text: {case['id']}.{key}")


def render() -> str:
    rules = json.loads((ROOT / 'configs/protocol/protocol_rules.json').read_text(encoding='utf-8'))
    result = subprocess.run(
        [sys.executable, '-B', str(ROOT / 'scripts/run_protocol_replay.py'),
         '--resolve-case-set', str(ROOT / 'configs/protocol/protocol_replay_cases.json')],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    resolved = json.loads(result.stdout)
    validate_case_copy(resolved['case_results'])
    data = {
        'protocol_version': rules['protocol_version'],
        'cases': resolved['case_results'],
        'rules': [{key: rule[key] for key in ('id', 'title_en', 'title_zh', 'action', 'priority', 'kind', 'predicate')}
                  for rule in rules['rules']],
        'resolver': rules['resolver'],
    }
    return 'window.PERSONAGUARD_CASES = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check without writing')
    args = parser.parse_args()
    try:
        expected = render()
    except (ValueError, KeyError, OSError) as error:
        print(f'Project-page data check failed: {error}', file=sys.stderr)
        return 1
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding='utf-8') != expected:
            print('Project-page case data is stale; run scripts/build_project_page_data.py.', file=sys.stderr)
            return 1
        print('Project-page data and presentation decisions match the public protocol resolver.')
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(expected, encoding='utf-8')
    print(f'Wrote {OUTPUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
