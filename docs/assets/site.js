(() => {
  'use strict';
  const repo = 'https://github.com/rudykon/PersonaGuard';
  const research = window.PERSONAGUARD_CASES;
  const caseButtons = [...document.querySelectorAll('[data-case]')];
  let selectedCase = 'optional_sensing';
  let language = 'en';
  let copyTimer;

  function renderCase() {
    if (!research) return;
    const record = research.cases.find(item => item.id === selectedCase);
    const rule = research.rules.find(item => item.id === record.matched_rule_id);
    document.getElementById('case-title').textContent = record[`label_${language}`];
    document.getElementById('case-use').textContent = record[`proposed_use_${language}`];
    document.getElementById('case-evidence').textContent = record[`evidence_basis_${language}`];
    document.getElementById('case-action').textContent = record[`audited_decision_${language}`];
    document.getElementById('case-rule-title').textContent = rule[`title_${language}`];
    const ruleId = rule.id.split('_')[0];
    document.getElementById('case-rule').textContent = ruleId;
    document.getElementById('case-source').href = `${repo}/blob/main/${record.source_locator}`;
    document.getElementById('case-source').setAttribute('aria-label', language === 'zh' ? '查看案例证据' : 'View case evidence');
    document.getElementById('case-panel').setAttribute('aria-labelledby', `tab-${selectedCase}`);
    caseButtons.forEach(button => {
      const active = button.dataset.case === selectedCase;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('[data-rule]').forEach(item => item.classList.toggle('active', item.dataset.rule === ruleId));
  }

  function setLanguage(next) {
    language = next === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en][data-zh]').forEach(node => { node.textContent = node.dataset[language]; });
    document.querySelectorAll('[data-lang]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.lang === language)));
    document.title = language === 'zh' ? 'PersonaGuard — 从证据到行动' : 'PersonaGuard — From Evidence to Action';
    document.querySelector('nav').setAttribute('aria-label', language === 'zh' ? '主导航' : 'Main navigation');
    document.querySelector('[role=tablist]').setAttribute('aria-label', language === 'zh' ? '已记录的研究案例' : 'Recorded research cases');
    clearTimeout(copyTimer);
    document.getElementById('copy-status').textContent = '';
    renderCase();
    try { localStorage.setItem('personaguard-language', language); } catch (_) { /* Language remains usable without storage. */ }
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
    copyTimer = setTimeout(() => { label.textContent = language === 'zh' ? '复制' : 'Copy'; }, 3500);
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        document.querySelectorAll('.desktop-nav a').forEach(link => {
          if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    }, { rootMargin: '-15% 0px -65% 0px' });
    ['overview', 'protocol', 'evidence', 'resources'].forEach(id => observer.observe(document.getElementById(id)));
  }
  try { language = localStorage.getItem('personaguard-language') === 'zh' ? 'zh' : 'en'; } catch (_) { /* Use English by default. */ }
  setLanguage(language);
})();
