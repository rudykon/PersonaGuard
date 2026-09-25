"""Build bilingual research pages from one reviewed set of HTML fragments."""
from pathlib import Path
import json
import posixpath
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from mkdocs.structure.files import File
from mkdocs.plugins import event_priority
from mkdocs.structure.toc import AnchorLink, TableOfContents

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / 'website/overrides/content'
PUBLIC_ASSETS = ('cases.js', 'case-copy.js', 'audit-data.js', 'audit-engine.js', 'mark.svg')


def on_files(files, config, **kwargs):
    # Reviewed public assets remain where the data generators and parity tests
    # expect them. Register only this allowlist; never copy the research workspace.
    for file in list(files):
        if file.src_uri == 'hooks.py' or file.src_uri.startswith('overrides/') or '__pycache__' in Path(file.src_uri).parts or file.src_uri.endswith('.pyc'):
            files.remove(file)
    assets = [ROOT / 'docs/assets' / name for name in PUBLIC_ASSETS]
    assets += sorted((ROOT / 'docs/assets/figures').glob('*.webp'))
    for path in assets:
        relative = path.relative_to(ROOT / 'docs').as_posix()
        if files.get_file_from_path(relative) is None:
            files.append(File(relative, str(ROOT / 'docs'), config.site_dir, config.use_directory_urls))
    return files


def on_page_markdown(markdown, page, config, **kwargs):
    fragment = page.meta.get('fragment')
    if not fragment:
        return markdown
    language = config.theme['language']
    language = 'zh' if language.startswith('zh') else 'en'
    page.meta['title'] = page.meta.get('title_zh' if language == 'zh' else 'title', page.title)
    page.title = page.meta['title']
    soup = BeautifulSoup((CONTENT / f'{fragment}.html').read_text(encoding='utf-8'), 'html.parser')
    for node in soup.select('[data-en][data-zh]'):
        node.string = node[f'data-{language}']
    for attr in ('alt', 'label', 'title'):
        for node in soup.select(f'[data-{attr}-{language}]'):
            node['aria-label' if attr == 'label' else attr] = node[f'data-{attr}-{language}']
    if fragment == 'results':
        cases = json.loads((ROOT / 'docs/assets/cases.js').read_text().split(' = ', 1)[1].strip().removesuffix(';'))
        copy = json.loads((ROOT / 'docs/assets/case-copy.js').read_text().split(' = ', 1)[1].strip().removesuffix(';'))
        record = next(item for item in cases['cases'] if item['id'] == 'trace_interpretation')
        for field, element in {'proposed_use': 'case-official-use', 'evidence_basis': 'case-official-evidence', 'audited_decision': 'case-official-action'}.items():
            soup.find(id=element).string = record[f'{field}_{language}']
        for node in soup.select('[data-summary-case]'):
            node.string = copy[node['data-summary-case']][node['data-summary-field']][language]
    # data-page and data-asset are locale-aware and independent of the Pages prefix.
    prefix = 'zh/' if language == 'zh' else ''
    current = page.url.rstrip('/') if page.url.endswith('/') else posixpath.dirname(page.url)
    for node in soup.select('[data-page]'):
        route = urlsplit(node['data-page'])
        destination = prefix + route.path
        node['href'] = posixpath.relpath(destination or '.', current or '.')
        if not node['href'].endswith('/'):
            node['href'] += '/'
        if route.query:
            node['href'] += '?' + route.query
        if route.fragment:
            node['href'] += '#' + route.fragment
    for node in soup.select('[data-asset]'):
        node['src' if node.name == 'img' else 'href'] = posixpath.relpath(node['data-asset'], current or '.')
    # A small per-page TOC also works for the original HTML figures and tables.
    return markdown + '\n\n' + str(soup)


def on_page_content(html, page, **kwargs):
    soup = BeautifulSoup(html, 'html.parser')
    headings = []
    for index, heading in enumerate(soup.select('h2, h3')):
        if heading.find_parent(['details', 'figure']) or heading.get('id', '').startswith('case-'):
            continue
        anchor = heading.get('id') or f'section-{index + 1}'
        heading['id'] = anchor
        item = AnchorLink(heading.get_text(' ', strip=True), anchor, int(heading.name[1]))
        if heading.name == 'h3' and headings:
            headings[-1].children.append(item)
        else:
            headings.append(item)
    if headings:
        page.toc = TableOfContents(headings)
    return str(soup)


@event_priority(-200)
def on_post_build(config, **kwargs):
    # Preserve old bookmarks and case query strings used by the public README.
    output = Path(config.site_dir)
    (output / '.nojekyll').touch()
    for locale in ('', 'zh/'):
        destination = output / locale / 'playground.html'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>PersonaGuard browser demo</title>'
            '<script>location.replace("demo/" + location.search + location.hash)</script>'
            '<p><a href="demo/">Open browser demo / 打开浏览器演示</a></p></html>', encoding='utf-8')

    # Material 9.7 discovers sitemaps relative to each alternate chapter URL.
    # The i18n plugin provides chapter-level hreflang links, so expose aliases
    # of the canonical sitemap at those locations without patching the theme.
    sitemap = output / 'sitemap.xml'
    if sitemap.exists():
        for locale in ('', 'zh/'):
            for route in ('', 'method/', 'evaluation/', 'results/', 'discussion/', 'demo/', 'reproduce/'):
                alias = output / locale / route / 'sitemap.xml'
                if alias != sitemap:
                    alias.parent.mkdir(parents=True, exist_ok=True)
                    alias.write_bytes(sitemap.read_bytes())
