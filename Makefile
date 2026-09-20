PYTHON ?= python3
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := src

.PHONY: help check github-check github-export smoke publication-check paper

help:
	@echo "make check              Offline repository and export tests (standard library)"
	@echo "make github-check       Read-only Git candidate audit"
	@echo "make github-export      Export dist/CHI2027-github.zip without local files or Git history"
	@echo "make smoke              Synthetic structural smoke (requires NumPy)"
	@echo "make publication-check  Local publication check (requires manuscripts and artifacts)"
	@echo "make paper              Compile local English and Chinese manuscripts"
	@echo "Override interpreter:   make smoke PYTHON=.venv/bin/python"

check:
	$(PYTHON) -B -m unittest -v tests.test_repository_layout tests.test_export_github

github-check:
	$(PYTHON) -B scripts/check_repository.py

github-export:
	$(PYTHON) -B scripts/export_github.py

smoke:
	$(PYTHON) -B scripts/run_revision6_synthetic_smoke.py

publication-check:
	$(PYTHON) -B scripts/build_revision6_publication.py --check

paper:
	mkdir -p artifacts/paper_build/main artifacts/paper_build/si artifacts/paper_build/main_zh artifacts/paper_build/si_zh
	cd paper && latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/main main.tex
	cd paper && latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/si supplementary_information.tex
	cd paper_zh && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/main_zh main.tex
	cd paper_zh && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../artifacts/paper_build/si_zh supplementary_information.tex
