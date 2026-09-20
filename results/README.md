# Results and evidence records

This directory contains the three maintained aggregate records consumed by the analysis summaries and figure generators.

| File | Purpose | Update entry point |
|---|---|---|
| `revision6_source.json` | Canonical aggregate numerical source, including upstream paths and SHA-256 hashes | `scripts/build_revision6_source.py` |
| `evidence_traceability.json` | Evidence graph, scope boundaries, decisions, and artifact registry | `scripts/refresh_evidence_traceability.py`; validate with `scripts/generate_evidence_traceability.py` |
| `statistical_analysis_registry.json` | Statistical reporting registry derived from the canonical source | `scripts/generate_revision6_publication.py` |

Protocol definitions, worked cases, external reuse cases, and comparison records live in [configs/protocol](../configs/protocol/README.md).
Figure CSVs and the authorized anonymous trajectory examples live with the [plotting code](../scripts/figures/README.md).

Run this read-only check from the repository root when the recorded local analysis inputs are available:

```bash
.venv/bin/python -B scripts/build_revision6_source.py --check
```

Rebuilding the aggregate source requires the upstream analysis outputs and feature manifests described in the [reproduction guide](../docs/REPRODUCING.md).
Those local inputs remain in `artifacts/` and the configured data directories.
Moving records or refreshing paths and hashes does not constitute a new empirical analysis.
