# PersonaGuard

**A Companion for Deciding When AI Should Personalize**

**Auditing Personalization Decisions in HCI Systems**

**English** · [简体中文](README.zh-CN.md)

An executable evidence-to-action audit protocol and reproducible HCI research code.

This project asks **which personalization decisions the available evidence can support, and where those decisions should stop**.
The protocol connects evidence types, evaluation conditions, and decision rules across offline estimation,
optional physiological sensing, behavioral calibration, and profile retention or transfer.
It records when personalization is permitted within an evaluated boundary, when an evaluated comparator
should be retained, and when further evidence or a user study is needed.

Continuous valence–arousal data provide a worked example for the protocol.
Existing checks support structural conformance,
representation of evidence from multiple sources, and deterministic replay. Independent analyst reuse,
protocol usability, and user benefit remain to be evaluated.

## 🗺️ Find your way around

| Task | Start here |
|---|---|
| Set up an environment or reproduce analyses | [Reproduction guide](docs/REPRODUCING.md) |
| Find the command for a particular analysis | [Script index](scripts/README.md) |
| Inspect protocol rules and audit cases | [Protocol rules](paper_support/protocol_rules.json) · [Replay cases](paper_support/protocol_replay_cases.json) |
| Find documentation for the current workflows | [Documentation index](docs/README.md) |
| Prepare a GitHub release | [Release checklist](docs/GITHUB.md) |

The detailed guides linked above are currently written in Chinese.

## 🚀 Quick start

Run these commands from the repository root using **Python 3.10+**.
The repository checks use only the Python standard library and require no GPU, research data, or network access.

```bash
# Run self-contained repository checks
python3 -B -m unittest -v tests.test_repository_layout

# Audit Git candidate files and local links without modifying them
python3 -B scripts/check_repository.py
```

The audit reads working-tree files. It does not change the Git index, commit, or upload files.
With GNU Make, `make github-check` runs the same audit.

With a NumPy-enabled research environment, run the synthetic structural smoke test:

```bash
make smoke PYTHON=.venv/bin/python
```

This checks grouped folds, threshold summaries, and evidence-graph structure using synthetic data.
It does not reproduce empirical findings. Full analyses require their authorized inputs and caches;
choose the relevant workflow from the [script index](scripts/README.md).

## 🗂️ Repository and evidence flow

```text
CHI2027/
├── paper_support/  Aggregate results, protocol rules and cases, figure sources, source-data CSVs
├── src/merps/      Reusable algorithms and models
├── scripts/        Analysis, generation, validation, and packaging entry points
├── tests/          Unit tests and research consistency checks
├── configs/        Experiment settings and stimulus manifests
├── docs/           Current workflow guides and release documentation
├── data/           Local data and caches; only selected documentation is versioned
├── checkpoints/    Local model weights
├── artifacts/      Local experiment results, audit reports, and build outputs
└── .local/         Machine-local credentials and handoff notes
```

Aggregate results and audit decisions follow one evidence chain:

```text
Research inputs and formal analyses
                ↓
paper_support/revision6_source.json
                ↓
Aggregate summaries, source-data tables, and audit outputs
```

The maintained code covers protocol replay, feature preparation, content priors, physiological
residuals, sparse feedback, and measurement audits. See the [script index](scripts/README.md)
for the entry point and input requirements of each workflow.

## 🔧 Environment and sharing

`requirements.txt` preserves the full research environment constraints, including pinned CUDA/PyTorch
versions. `pyproject.toml` provides the local package installation entry point. Lightweight checks
can run without installing every training dependency. See the [reproduction guide](docs/REPRODUCING.md)
for data authorization, environment requirements, and runtime costs.

- **Local materials:** raw data, stimulus videos, complete participant-level outputs, model weights,
  credentials, and backups are excluded from ordinary repository distribution.
- **Supporting materials:** check the provenance and applicable permissions for aggregate data,
  figure sources, and anonymous examples.
- **License:** no repository-wide open-source license has been selected. Read access does not grant
  redistribution rights.
- **Release preparation:** review Git history, third-party licenses, and anonymous-review identity
  separately, following the [release checklist](docs/GITHUB.md).

## 📊 Results and provenance

The detailed result ledger is generated from
[`paper_support/revision6_source.json`](paper_support/revision6_source.json), the sole numerical source
for aggregate results, source-data tables, and audit summaries. Read the
[generated result summary (Chinese)](README.zh-CN.md#results) for the recorded findings and their scope.

Results are interpreted within each analysis's information conditions and comparator.
Post-trial sparse recovery uses feedback collected after a trial. A residual model that has not
demonstrated incremental value does not establish equivalence or show that physiology contains no
information. When evidence for increment, stability, outcomes, or governance is insufficient, the
protocol records the resulting constraint or abstention.
