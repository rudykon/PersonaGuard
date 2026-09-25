#!/usr/bin/env python3
"""Check built bilingual pages, local links, and assets without network access."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.links, self.h1 = set(), [], 0
        self.language = ''
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'html':
            self.language = attrs.get('lang', '')
        if tag == 'h1':
            self.h1 += 1
        for key in ('src', 'href'):
            if attrs.get(key):
                self.links.append(attrs[key])


def check(root):
    pages = {path.resolve(): Page(path) for path in root.rglob('*.html')}
    errors = []
    for locale in ('', 'zh/'):
        for route in ('', 'method/', 'evaluation/', 'results/', 'discussion/', 'demo/', 'reproduce/'):
            path = (root / locale / route / 'index.html').resolve()
            page = pages.get(path)
            if not page:
                errors.append(f'Missing page: {locale}{route}')
            elif page.h1 != 1 or not page.language.startswith('zh' if locale else 'en'):
                errors.append(f'Invalid page title/language: {locale}{route} ({page.h1} H1s, {page.language})')
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = unquote(url.path)
            if target.startswith('/PersonaGuard/'):
                destination = root / target.removeprefix('/PersonaGuard/')
            elif target.startswith('/'):
                destination = root / target.lstrip('/')
            else:
                destination = path.parent / target if target else path
            if destination.is_dir():
                destination /= 'index.html'
            destination = destination.resolve()
            if not destination.exists():
                errors.append(f'{path.relative_to(root)}: missing {link}')
            elif url.fragment and destination in pages and unquote(url.fragment) not in pages[destination].ids:
                errors.append(f'{path.relative_to(root)}: missing anchor {link}')
    for error in sorted(set(errors)):
        print(error)
    if errors:
        return 1
    print(f'Checked {len(pages)} HTML pages: bilingual routes, one H1 per page, local links, anchors, and assets pass.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('dist/project-site'))
    args = parser.parse_args()
    raise SystemExit(check(args.directory.resolve()))
