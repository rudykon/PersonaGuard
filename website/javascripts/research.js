(() => {
  'use strict';
  // i18n already supplies the exact translated chapter; bypass the theme's
  // language-root heuristic while retaining native link and keyboard behavior.
  document.addEventListener('click', event => {
    if (event.target instanceof Element && event.target.closest('.md-select__link')) event.stopPropagation();
  }, true);
  const repo = 'https://github.com/rudykon/PersonaGuard';
  const research = window.PERSONAGUARD_CASES;
  const copy = window.PERSONAGUARD_CASE_COPY;
  const caseButtons = [...document.querySelectorAll('[data-case]')];
  let selectedCase = 'trace_interpretation';
  let language = 'en';
  let copyTimer;

  function renderCase() {
    if (!research || !copy || !document.getElementById('case-panel')) return;
    const record = research.cases.find(item => item.id === selectedCase);
    const explanation = copy[selectedCase];
    document.getElementById('case-run-link').href = `../demo/?case=${encodeURIComponent(selectedCase)}`;
    const rule = research.rules.find(item => item.id === record.matched_rule_id);
    if (!explanation || explanation.expected_action !== record.audited_action) {
      throw new Error(`Presentation copy must be reviewed for ${selectedCase}`);
    }
    for (const [field, id] of Object.entries({
      question: 'case-question', use: 'case-use', evidence: 'case-evidence',
      next: 'case-action', why: 'case-why', context: 'case-context', review: 'case-review',
    })) document.getElementById(id).textContent = explanation[field][language];
    for (const [field, id] of Object.entries({
      proposed_use: 'case-official-use', evidence_basis: 'case-official-evidence',
      audited_decision: 'case-official-action',
    })) document.getElementById(id).textContent = record[`${field}_${language}`];
    document.getElementById('case-rule').textContent = rule.id.split('_')[0];
    document.getElementById('case-rule-title').textContent = rule[`title_${language}`];
    document.getElementById('case-priority').textContent = rule.priority;
    document.getElementById('case-source').href = `${repo}/blob/main/${record.source_locator.split('#')[0]}`;
    const conditions = document.getElementById('case-conditions');
    conditions.replaceChildren();
    for (const [field, value] of Object.entries(record.decisive_values)) {
      const row = document.createElement('tr');
      for (const text of [field, value]) {
        const cell = document.createElement('td');
        cell.textContent = String(text);
        row.append(cell);
      }
      conditions.append(row);
    }
    const ownerNames = {
      'measurement-analysis lead': { en: 'Measurement-analysis lead', zh: '测量分析负责人' },
      'sensor-analysis lead': { en: 'Sensor-analysis lead', zh: '传感分析负责人' },
      'personalization-analysis lead': { en: 'Personalization-analysis lead', zh: '个性化分析负责人' },
      'data-governance lead': { en: 'Data-governance lead', zh: '数据治理负责人' },
    };
    document.getElementById('case-owner').textContent = ownerNames[record.decision_owner_role]?.[language] || record.decision_owner_role;
    document.getElementById('case-review-trigger').textContent = record.review_trigger;
    document.getElementById('case-panel').setAttribute('aria-labelledby', `tab-${selectedCase}`);
    caseButtons.forEach(button => {
      const active = button.dataset.case === selectedCase;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
  }

  function renderRuleOrder() {
    if (!research || !document.getElementById('rule-order')) return;
    document.getElementById('rule-order').textContent = [...research.rules]
      .sort((a, b) => a.priority - b.priority)
      .map(rule => rule.id.split('_')[0]).join(' → ');
  }

  function setLanguage(next) {
    language = next === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en][data-zh]').forEach(node => { node.textContent = node.dataset[language]; });
    document.querySelectorAll('[data-lang]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.lang === language)));
    document.querySelectorAll('meta[data-content-en][data-content-zh]').forEach(node => { node.content = node.getAttribute(`data-content-${language}`); });
    document.querySelectorAll('[data-alt-en][data-alt-zh]').forEach(node => { node.alt = node.getAttribute(`data-alt-${language}`); });
    document.querySelectorAll('[data-label-en][data-label-zh]').forEach(node => { node.setAttribute('aria-label', node.getAttribute(`data-label-${language}`)); });
    document.querySelectorAll('[data-summary-case]').forEach(node => {
      const item = copy?.[node.dataset.summaryCase];
      if (item) node.textContent = item[node.dataset.summaryField][language];
    });
    document.querySelectorAll('[data-number]').forEach(node => {
      const value = research?.metrics?.[node.dataset.number];
      if (value !== undefined) node.textContent = value;
    });
    document.querySelector('[role=tablist]')?.setAttribute('aria-label', language === 'zh' ? '已有审查示例' : 'Recorded review examples');
    clearTimeout(copyTimer);
    if (document.getElementById('copy-status')) document.getElementById('copy-status').textContent = '';
    renderCase();
    renderRuleOrder();
    try { localStorage.setItem('personaguard-language', language); } catch (_) { /* The page remains usable without storage. */ }
  }

  document.querySelectorAll('[data-lang]').forEach(button => button.addEventListener('click', () => setLanguage(button.dataset.lang)));
  caseButtons.forEach((button, index) => {
    button.addEventListener('click', () => { selectedCase = button.dataset.case; renderCase(); });
    button.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (index + 1) % caseButtons.length;
      if (event.key === 'ArrowLeft') target = (index + caseButtons.length - 1) % caseButtons.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = caseButtons.length - 1;
      if (target === undefined) return;
      event.preventDefault();
      selectedCase = caseButtons[target].dataset.case;
      renderCase();
      caseButtons[target].focus();
    });
  });

  document.getElementById('copy-code')?.addEventListener('click', async () => {
    const code = document.getElementById('quickstart-code');
    const status = document.getElementById('copy-status');
    const label = document.querySelector('#copy-code [data-en]');
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(code.textContent);
      label.textContent = language === 'zh' ? '已复制' : 'Copied';
      status.textContent = language === 'zh' ? '命令已复制到剪贴板。' : 'Commands copied to clipboard.';
    } catch (_) {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      label.textContent = language === 'zh' ? '按 Ctrl/Cmd+C' : 'Press Ctrl/Cmd+C';
      status.textContent = language === 'zh' ? '已选中命令，请手动复制。' : 'Commands selected. Copy with Ctrl+C or Command+C.';
    }
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { label.textContent = language === 'zh' ? '复制命令' : 'Copy commands'; }, 3500);
  });

  setLanguage(document.documentElement.lang.startsWith('zh') ? 'zh' : 'en');
  if (research && copy) document.documentElement.classList.add('js-ready');
  // Preserve detailed bookmarks from the original single-page site.
  if (document.querySelector('.home-hero')) {
    const anchor = location.hash.slice(1);
    const routes = {evaluation: 'evaluation/', discussion: 'discussion/', protocol: 'method/#protocol', evidence: 'results/', examples: 'results/#examples'};
    const route = routes[anchor] || (/^(result-|fig-[3-5]|scenario-results)/.test(anchor) ? `results/#${anchor}` : /^rule-R/.test(anchor) ? `method/#${anchor}` : anchor === 'fig-1' ? 'method/#fig-1' : anchor === 'fig-2' ? 'evaluation/#fig-2' : '');
    if (route) location.replace(new URL(route, location.href));
  }
})();
