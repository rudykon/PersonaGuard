# Protocol definitions and cases

These JSON files provide the maintained definitions, worked examples, and comparison records for the audit protocol.

| File | Purpose |
|---|---|
| `protocol_rules.json` | Declared evidence fields, actions, and deterministic decision rules |
| `protocol_replay_cases.json` | Four worked route records; source locators point into `results/evidence_traceability.json` |
| `external_reuse_cases.json` | Seven routes from six external sources, including source coding and scope |
| `adjacent_frameworks.json` | Structured comparison with adjacent frameworks and comparison boundaries |

Resolve the worked case set with the maintained rule file, from the repository root:

```bash
python3 -B scripts/run_protocol_replay.py --resolve-case-set configs/protocol/protocol_replay_cases.json
```

Resolve the external cases with their DOI-anchored source records:

```bash
python3 -B scripts/run_protocol_replay.py --resolve-case-set configs/protocol/external_reuse_cases.json --external-case-set
```

These commands use only the Python standard library. Each command checks the case schema and applies the declared rules. It does not rerun empirical analyses or independently validate the evidence coding.
Aggregate results and the evidence graph are documented in [results](../../results/README.md).
