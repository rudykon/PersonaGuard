#!/usr/bin/env python3
"""Export public rule inputs and validated examples for the browser demo."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import run_protocol_replay as replay

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/assets/audit-data.js'
PREFIX = 'window.PERSONAGUARD_AUDIT_DATA = '


def render() -> str:
    rule_path = ROOT / 'configs/protocol/protocol_rules.json'
    protocol = replay.read_json(rule_path)
    replay.validate_protocol(protocol)
    files = [rule_path]
    samples = []
    for name, external in [('protocol_replay_cases.json', False), ('external_reuse_cases.json', True)]:
        path = ROOT / 'configs/protocol' / name
        specification = replay.read_json(path)
        replay.validate_case_set(specification, protocol, external=external)
        files.append(path)
        for record in specification['cases']:
            samples.append({
                'record': record,
                'group': 'external' if external else 'worked',
                'source_path': path.relative_to(ROOT).as_posix(),
                'baseline': replay.evaluate_case(record, protocol),
            })
    data = {
        'schema_version': 'personaguard-browser-inputs-v1',
        'protocol': protocol,
        'stages': sorted(replay.ALLOWED_STAGES),
        'capabilities': sorted(replay.ALLOWED_CAPABILITIES),
        'forbidden_case_fields': sorted(replay.FORBIDDEN_CASE_FIELDS),
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'samples': samples,
    }
    return PREFIX + json.dumps(data, ensure_ascii=False, indent=2) + ';\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding='utf-8') != expected:
            print('Browser inputs are stale; run scripts/build_browser_demo_data.py.')
            return 1
        print('Browser rules and all 11 examples match the public Python resolver.')
    else:
        OUTPUT.write_text(expected, encoding='utf-8')
        print('Wrote docs/assets/audit-data.js')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
