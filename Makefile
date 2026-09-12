PYTHON ?= python3
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := src

.PHONY: help check github-check smoke publication-check paper

help:
	@echo "make check              Offline repository tests (Python standard library)"
	@echo "make github-check       Read-only Git candidate safety audit"
	@echo "make smoke              Synthetic structural smoke (requires NumPy)"
	@echo "make publication-check  Full local-workspace publication check (requires artifacts)"
	@echo "make paper              Compile four manuscripts into artifacts/paper_build/"
	@echo "Override interpreter:   make smoke PYTHON=.venv/bin/python"

check:
	$(PYTHON) -B -m unittest -v tests.test_repository_layout

github-check:
	$(PYTHON) -B scripts/check_repository.py

smoke:
	$(PYTHON) -B scripts/run_revision6_synthetic_smoke.py

publication-check:
	$(PYTHON) -B scripts/build_revision6_publication.py --check

paper:
	mkdir -p artifacts/paper_build/main artifacts/paper_build/si artifacts/paper_build/main_zh artifacts/paper_build/si_zh
	cd paper && latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/main main.tex
	cd paper && latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/si supplementary_information.tex
	cd paper && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/main_zh main_zh.tex
	cd paper && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/si_zh supplementary_information_zh.tex
