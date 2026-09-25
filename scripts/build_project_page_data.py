#!/usr/bin/env python3
"""Build the project page's recorded cases from the public protocol resolver."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

from generate_revision6_publication import macros

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
        for key in ('question', 'use', 'evidence', 'next', 'why', 'context', 'review', 'summary_use', 'summary_boundary', 'summary_action'):
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
    graph = json.loads((ROOT / 'results/evidence_traceability.json').read_text(encoding='utf-8'))
    nodes = {node['id']: node for node in graph['nodes']}
    for case in resolved['case_results']:
        node = nodes[case['source_locator'].split('#', 1)[1]]
        # These are provenance fields, not new predicates or evidence judgments.
        case['decision_owner_role'] = node['decision_owner_role']
        case['review_trigger'] = node['review_trigger']
    source = json.loads((ROOT / 'results/revision6_source.json').read_text(encoding='utf-8'))
    metrics = {}
    for line in macros(source).splitlines():
        match = re.fullmatch(r'\\newcommand\{\\(\w+)\}\{(.*)\}', line)
        if match:
            value = match[2].replace(r'\(', '').replace(r'\)', '').replace(r'\%', '%')
            value = re.sub(r'(?<![\d.])\.(\d)', r'0.\1', value)
            metrics[match[1]] = value.replace('--', '–')
    metrics['PathMeanProfileRank'] = f"{source['summaries']['revision4']['dtw_objective_audit']['objectives']['path_mean']['profile_spearman_vs_direct']:.3f}"
    metrics['ReferenceMaxShift'] = f"{source['summaries']['revision5']['reference_ruler_audit']['common_panel_comparison']['maximum_profile_shift_seconds']:.3f}"
    for budget, values in source['summaries']['revision5']['signed_calibration_practical_value']['budgets'].items():
        metrics[f'SignedGainBudget{budget}'] = f"{values['calibrated_gain_vs_video_units']:.3f}"
    data = {
        'protocol_version': rules['protocol_version'],
        'cases': resolved['case_results'],
        'rules': [{key: rule[key] for key in ('id', 'title_en', 'title_zh', 'action', 'priority', 'kind', 'predicate')}
                  for rule in rules['rules']],
        'resolver': rules['resolver'],
        'metrics': metrics,
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
    metrics = json.loads(expected.split(' = ', 1)[1].strip().removesuffix(';'))['metrics']
    # Reviewed bilingual fragments are the source; docs/ contains built pages.
    pages = sorted((ROOT / 'website/overrides/content').glob('*.html'))
    if not pages:
        pages = [ROOT / 'docs/index.html']
    number_pattern = r'<span data-number="(\w+)">([^<]*)</span>'
    for path in pages:
        page = path.read_text(encoding='utf-8')
        stale = [key for key, value in re.findall(number_pattern, page) if metrics.get(key) != value]
        if args.check and stale:
            print(f'Project-page numbers are stale in {path.name}: {sorted(set(stale))}', file=sys.stderr)
            return 1
        if not args.check:
            page = re.sub(number_pattern, lambda match: f'<span data-number="{match[1]}">{metrics[match[1]]}</span>', page)
            path.write_text(page, encoding='utf-8')
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
