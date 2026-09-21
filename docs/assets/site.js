(() => {
  'use strict';
  const repo = 'https://github.com/rudykon/PersonaGuard';
  const research = window.PERSONAGUARD_CASES;
  const copy = window.PERSONAGUARD_CASE_COPY;
  const caseButtons = [...document.querySelectorAll('[data-case]')];
  let selectedCase = 'trace_interpretation';
  let language = 'en';
  let copyTimer;

  function renderCase() {
    if (!research || !copy) return;
    const record = research.cases.find(item => item.id === selectedCase);
    const explanation = copy[selectedCase];
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
    if (!research) return;
    document.getElementById('rule-order').textContent = [...research.rules]
      .sort((a, b) => a.priority - b.priority)
      .map(rule => rule.id.split('_')[0]).join(' → ');
  }

  function setLanguage(next) {
    language = next === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en][data-zh]').forEach(node => { node.textContent = node.dataset[language]; });
    document.querySelectorAll('[data-lang]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.lang === language)));
    document.title = 'PersonaGuard — From Evidence to Action';
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
    document.querySelector('nav').setAttribute('aria-label', language === 'zh' ? '主导航' : 'Main navigation');
    document.querySelector('[role=tablist]').setAttribute('aria-label', language === 'zh' ? '已有审查示例' : 'Recorded review examples');
    clearTimeout(copyTimer);
    document.getElementById('copy-status').textContent = '';
    renderCase();
    renderRuleOrder();
    scheduleSectionUpdate();
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

  document.getElementById('copy-code').addEventListener('click', async () => {
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

  const sections = ['overview', 'method', 'evaluation', 'results', 'discussion', 'resources']
    .map(id => document.getElementById(id));
  let scrollFrame;
  function markCurrentSection() {
    scrollFrame = undefined;
    const threshold = document.querySelector('.header').getBoundingClientRect().bottom + 30;
    let current = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= threshold) current = section;
    }
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
      current = sections[sections.length - 1];
    }
    document.querySelectorAll('.main-nav a').forEach(link => {
      if (link.hash === `#${current.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  function scheduleSectionUpdate() {
    if (scrollFrame === undefined) scrollFrame = requestAnimationFrame(markCurrentSection);
  }
  window.addEventListener('scroll', scheduleSectionUpdate, { passive: true });
  window.addEventListener('resize', scheduleSectionUpdate);
  window.addEventListener('load', scheduleSectionUpdate);
  scheduleSectionUpdate();
  try { language = localStorage.getItem('personaguard-language') === 'zh' ? 'zh' : 'en'; } catch (_) { /* Use English by default. */ }
  setLanguage(language);
  if (research && copy) document.documentElement.classList.add('js-ready');
})();
