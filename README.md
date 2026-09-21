<p align="center">
  <img src="docs/brand-mark.svg" width="520" alt="PersonaGuard brand mark">
</p>

<h1 align="center">PersonaGuard</h1>

<p align="center">
  <strong>A Companion for Deciding When AI Should Personalize</strong><br>
  Auditing Personalization Decisions in HCI Systems
</p>

<p align="center">
  <a href="https://rudykon.github.io/PersonaGuard/"><img src="https://img.shields.io/badge/Project%20Website-Explore%20Research-28654a?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Open the PersonaGuard project website"></a>
  <a href="docs/REPRODUCING.md"><img src="https://img.shields.io/badge/Reproduction-Read%20Guide-3776AB?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Read the reproduction guide"></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="configs/protocol/README.md"><img src="https://img.shields.io/badge/Protocol-5%20Rules-28654a?style=flat-square" alt="View protocol rules and cases"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Validation-Structural%20Checks-2CA02C?style=flat-square" alt="Run repository structural checks"></a>
</p>

<p align="center">
  <a href="https://rudykon.github.io/PersonaGuard/">Website</a> ·
  <a href="https://rudykon.github.io/PersonaGuard/#method">Method</a> ·
  <a href="https://rudykon.github.io/PersonaGuard/#results">Results</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/REPRODUCING.md">Reproduction</a> ·
  <a href="#repository">Repository</a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

An executable protocol for deciding which personalization actions the available
evidence supports. Continuous valence–arousal estimation provides the worked
example, covering content priors, physiological sensing, sparse feedback, and
profile retention or transfer.

<a id="quick-start"></a>

## Quick start

Python 3.10+ and Git are required. From the repository root:

```bash
# Repository tests and file audit; no research data or GPU needed
make check github-check

# Resolve the included audit cases using the Python standard library
python3 -B scripts/run_protocol_replay.py --resolve-case-set configs/protocol/protocol_replay_cases.json
```

Without Make, use `python3 -B -m unittest -v tests.test_repository_layout tests.test_export_github`
and `python3 -B scripts/check_repository.py` for the checks.
For the NumPy-based synthetic smoke test and full research environment, see the
[reproduction guide](docs/REPRODUCING.md).

<a id="repository"></a>

## Repository

```text
src/merps/       Reusable algorithms
scripts/        Analyses, checks, exports, and figure source data
configs/        Protocol rules, cases, and stimulus manifests
results/        Aggregate results and evidence records
tests/          Unit tests and research consistency checks
docs/           Reproduction and GitHub instructions
data/           Data access and provenance documentation only
```

| Task | Guide |
|---|---|
| Install dependencies and reproduce analyses | [Reproduction](docs/REPRODUCING.md) |
| Find analysis and figure commands | [Script index](scripts/README.md) |
| Inspect rules and worked cases | [Protocol](configs/protocol/README.md) |
| Inspect results and provenance | [Results](results/README.md) |
| Export a clean source archive for GitHub | [GitHub guide](docs/GITHUB.md) |

Detailed workflow guides are currently in Chinese.

## Scope and availability

[`results/revision6_source.json`](results/revision6_source.json) is the canonical
aggregate numerical source. Current checks establish structural conformance and
deterministic replay; independent analyst reuse, usability, and user benefit
remain unevaluated. See the [recorded findings](README.zh-CN.md#results).

Raw data, weights, experiment caches, manuscript files, credentials, and backups
stay local. Full analyses require authorized data and the recorded upstream
artifacts. A repository-wide license has not yet been selected.
