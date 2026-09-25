#!/usr/bin/env python3
"""Build MkDocs, then update the existing main/docs GitHub Pages publication."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='build and validate without updating docs/')
    args = parser.parse_args()
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/build_project_page_data.py'), '--check'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-B', '-m', 'mkdocs', 'build', '--strict', '--clean'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_project_site.py')], cwd=ROOT, check=True)
    if not args.check:
        # Do not clean docs/: reproduction guides and generated research data
        # share this directory with the published site.
        shutil.copytree(ROOT / 'dist/project-site', ROOT / 'docs', dirs_exist_ok=True)
        print('Updated docs/ for GitHub Pages (main branch, /docs).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
