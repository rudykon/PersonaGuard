(() => {
  'use strict';
  if (!document.getElementById('run-form')) return;
  const data = window.PERSONAGUARD_AUDIT_DATA;
  const engine = window.PERSONAGUARD_AUDIT;
  const caseCopy = window.PERSONAGUARD_CASE_COPY || {};
  const $ = id => document.getElementById(id);
  const clone = value => JSON.parse(JSON.stringify(value));
  let language = document.documentElement.lang.startsWith('zh') ? 'zh' : 'en', sample, draft, lastRun, startupError, pending = false;
  const text = (en, zh) => language === 'zh' ? zh : en;
  const fields = {
    'evidence.interpretation_scope': ['Does the evidence cover this use?', '证据是否覆盖这项用途？', 'Check the tested people, task, interface, and information.', '核对已测试的人群、任务、界面与信息条件。'],
    'evidence.comparator_increment': ['Does it improve on the tested comparator?', '是否优于已测试的对照？', 'Use a comparison for the same task and available information.', '比较对象须匹配任务和当时可用的信息。'],
    'evidence.evaluation_mode': ['How was it evaluated?', '用什么方式评估？', 'A model metric and a study of people using the system are different evidence.', '模型指标与人实际使用系统的研究，是不同的证据。'],
    'evidence.user_outcome': ['What was observed for users?', '观察到了什么用户结果？', 'Distinguish a measurement improvement from task or experience outcomes.', '区分测量指标改善与任务、体验的实际结果。'],
    'deployment_stage': ['What use is being reviewed?', '当前审查哪种用途？', 'Changing this changes the proposed use, not just the strength of evidence.', '修改此项意味着改变用途，不只是补充证据。'],
    'route_capability': ['Can this route be implemented?', '这项方案能否实现？', '', ''],
    'evidence.user_control': ['Evidence of user control', '用户控制的证据', '', ''],
    'evidence.safety_reversibility': ['Can the change be reversed?', '调整能否撤回？', 'Reversibility alone does not establish user benefit.', '可以撤回，并不代表已证明用户受益。'],
    'evidence.persistence': ['Does it remain valid over time?', '以后是否仍然适用？', 'Relevant to long-term retention and reuse.', '用于审查长期保存与复用。'],
    'evidence.transfer': ['Does it apply in another setting?', '换个场景是否适用？', '', ''],
    'evidence.reference_equity': ['Does the reference cover the intended population?', '参考数据是否覆盖目标人群？', '', ''],
  };
  const values = {
    MATCHED: ['Matches the intended use', '与本用途匹配'], PARTIAL: ['Partial support', '仅部分支持'], NOT_TESTED: ['Not tested', '尚未检验'], NOT_APPLICABLE: ['Not applicable', '不适用'], SUPPORTED: ['Supported by evidence', '已有证据支持'], NO_DEMONSTRATED_INCREMENT: ['No demonstrated improvement', '未证明额外改善'],
    IN_CONTEXT_USER_STUDY: ['User study in context', '情境内用户研究'], TECHNICAL_AND_EMPIRICAL_EVALUATION: ['Technical and empirical evaluation', '技术与实证评估'], CASE_STUDY_IN_USE: ['Case study in actual use', '实际使用中的案例研究'], LONGITUDINAL_FIELD_USE: ['Longitudinal field use', '长期实地使用'], PROXY_ONLY: ['Proxy metrics only', '仅评估代理指标'], VIGNETTE_ONLY: ['Hypothetical scenarios only', '仅评估情境描述'],
    DIRECT_TASK_AND_EXPERIENCE: ['Direct task and experience outcomes', '直接观察任务与体验结果'], DIRECT_USER_EXPERIENCE: ['Direct user experience outcomes', '直接观察用户体验'], PROXIMAL_ONLY: ['Proximal measurement only', '仅有近端测量指标'], PREFERENCE_ONLY: ['Stated preference only', '仅有偏好表达'], NOT_REPORTED: ['Not reported', '未报告'], USER_CONTROLLED: ['Controlled by the user', '由用户控制'], REVERSIBLE: ['Reversible', '可以撤回'], NOT_REQUIRED: ['Not required for this use', '本用途不要求'], NOT_ASSESSABLE: ['Cannot be assessed', '尚无法评估'],
    OFFLINE_ESTIMATION: ['Offline measurement and interpretation', '离线测量与解释'], OPTIONAL_ACQUISITION: ['Add information or sensors', '额外采集信息或传感'], REVERSIBLE_PERSONALIZATION: ['Reversible personalization', '可逆的个性化调整'], CONSEQUENTIAL_PERSONALIZATION: ['Personalization with consequences', '会产生实际后果的个性化'], RETENTION_TRANSFER: ['Long-term retention or transfer', '长期保留或跨场景迁移'], AVAILABLE: ['Available', '可以实现'], NOT_AVAILABLE: ['Not available', '尚无法实现'],
  };
  const actions = {
    BOUND_TO_EVIDENCE_SCOPE: ['Limit interpretation and use', '限定解释与使用范围', 'The interpretation exceeds the tested scope. This result limits use; it is not permission to personalize.', '解释超出了已测试的范围。这一结果限定用途，不等于许可个性化。'],
    RETAIN_EVALUATED_COMPARATOR: ['Keep the tested comparator', '保留已测试的对照方案', 'The selected evidence does not demonstrate an added benefit over the matching comparator.', '所选证据没有证明相对匹配对照的额外收益。'],
    RUN_PREREGISTERED_USER_STUDY: ['First run a user study in context', '先开展情境内用户研究', 'For this deployment stage, proxy or untested outcomes do not establish direct user benefit. Run a preregistered study before deployment.', '对于这项部署用途，代理指标或尚未检验的结果不能证明用户直接受益。部署前需开展预注册研究。'],
    WITHHOLD_RETENTION_AND_TRANSFER: ['Hold off on long-term reuse', '暂不长期保留或跨场景复用', 'At least one requirement for persistence, transfer, or reference coverage remains unresolved.', '稳定性、场景迁移或参考人群覆盖中，至少一项要求仍未满足。'],
    PROCEED_WITHIN_EVALUATED_BOUNDARY: ['Proceed within the evaluated boundary', '在已评估边界内实施', 'The selected conditions meet R5, and no higher-priority constraint matches. Permission is limited to this recorded evidence boundary.', '所选条件满足 R5，且没有更高优先级约束命中。许可仅限于这份记录中的已评估边界。'],
    REQUIRE_ROUTE_SPECIFIC_EVIDENCE: ['Gather use-specific evidence', '补充针对这项用途的证据', 'None of the five rules matches. Removing a restriction does not automatically satisfy a permission rule.', '五条规则均未匹配。消除某项限制，不代表自动满足许可条件。'],
  };
  const rules = {R4:['Long-term reuse','长期复用'],R2:['Added benefit','额外收益'],R3:['User outcomes','用户结果'],R1:['Scope of use','使用范围'],R5:['Permission conditions','许可条件']};
  function pair(values) { return values[language === 'zh' ? 1 : 0]; }
  function get(record, path) { return path.split('.').reduce((node, key) => node[key], record); }
  function set(record, path, value) { const parts=path.split('.');const key=parts.pop();parts.reduce((node, part)=>node[part],record)[key]=value; }
  function changes(record) { return Object.keys(fields).filter(path=>get(record,path)!==get(sample.record,path)).map(path=>({field:path,from:get(sample.record,path),to:get(record,path)})); }
  function optionLabel(value, path) { return path==='route_capability'&&value==='PARTIAL'?text('Partly available','部分可实现'):pair(values[value]); }
  function allowed(path) { return path==='deployment_stage'?data.stages:path==='route_capability'?data.capabilities:data.protocol.evidence_fields[path.split('.')[1]]; }
  function createFields() {
    Object.entries(fields).forEach(([path, labels], index)=>{
      const wrap=document.createElement('div');wrap.className='lab-field';
      const id='field-'+path.replaceAll('.','-');const label=document.createElement('label');label.htmlFor=id;label.dataset.en=labels[0];label.dataset.zh=labels[1];
      const select=document.createElement('select');select.id=id;select.dataset.field=path;
      allowed(path).forEach(value=>{const option=document.createElement('option');option.value=value;select.append(option);});
      wrap.append(label,select);
      if(labels[2]){const help=document.createElement('p');help.id=id+'-help';help.dataset.en=labels[2];help.dataset.zh=labels[3];select.setAttribute('aria-describedby',help.id);wrap.append(help);}
      select.addEventListener('change',()=>{set(draft,path,select.value);pending=true;renderResult();});
      $(index<4?'main-fields':'extra-fields').append(wrap);
    });
  }
  function renderSample() {
    $('sample-description').textContent=text('Original proposal: ','原始用途：')+(caseCopy[sample.record.id]?.use[language] || sample.record['proposed_use_'+language]);
    const source=$('sample-source');source.replaceChildren(document.createTextNode(sample.group==='external'?text('Author-coded external study. ','作者编码的外部研究记录。'):text('MER-PS worked case. ','MER-PS 工作案例。')));
    const link=document.createElement('a');link.href='https://github.com/rudykon/PersonaGuard/blob/main/'+sample.source_path;link.textContent=text('Source record ↗','来源记录 ↗');source.append(link);
    $('try-user-study').hidden=sample.record.id!=='signed_calibration';$('try-hint').hidden=sample.record.id!=='signed_calibration';
    $('baseline-result').textContent=text('Original record: ','原始记录：')+sample.baseline.matched_rule_id.split('_')[0]+' · '+pair(actions[sample.baseline.audited_action]);
  }
  function syncControls() { document.querySelectorAll('[data-field]').forEach(node=>{node.value=get(draft,node.dataset.field);}); }
  function selectSample(id) {
    sample=data.samples.find(item=>item.record.id===id)||data.samples.find(item=>item.record.id==='signed_calibration');
    draft=clone(sample.record);$('sample-select').value=sample.record.id;syncControls();renderSample();run();
  }
  function renderResult() {
    $('run-result').hidden=pending||!lastRun;
    if(pending){$('run-status').textContent=text('Conditions changed. Run the review to update the result.','条件已修改，请运行审查更新结果。');return;}
    if(!lastRun)return;
    const {result,trace,record}=lastRun, changed=changes(record);
    $('run-status').textContent=text('Calculated in this browser.','已在当前浏览器完成计算。');
    $('run-mode').textContent=changed.length?text(`Hypothetical conditions · ${changed.length} changes`,`假设条件 · ${changed.length} 项修改`):text('Replay of the original conditions','原始条件重放');
    $('run-rule').textContent=result.matched_rule_id==='FALLBACK'?text('No rule matched','无规则匹配'):result.matched_rule_id.split('_')[0];
    $('run-action').textContent=pair(actions[result.audited_action]);
    $('run-explanation').textContent=actions[result.audited_action][language==='zh'?3:2];
    $('changes-box').hidden=!changed.length;$('changed-fields').replaceChildren();
    changed.forEach(item=>{const li=document.createElement('li');li.textContent=pair(fields[item.field])+': '+optionLabel(item.from,item.field)+' → '+optionLabel(item.to,item.field);$('changed-fields').append(li);});
    $('rule-trace').replaceChildren();
    trace.forEach(item=>{
      const li=document.createElement('li');li.dataset.selected=String(item.selected);li.dataset.rule=item.id;
      const heading=document.createElement('strong');heading.textContent=item.id.split('_')[0]+' · '+pair(rules[item.id.split('_')[0]]);
      const state=document.createElement('p');state.textContent=item.selected?text('Matched; determines this result.','命中，决定本次结果。'):item.matches?text('Also matches; a higher-priority rule takes precedence.','也满足条件，但优先处理前面的规则。'):text('Conditions not met.','未满足条件。');li.append(heading,state);
      const list=document.createElement('ul');Object.entries(item.values).forEach(([path,value])=>{const row=document.createElement('li');row.textContent=pair(fields[path])+': '+optionLabel(value,path);list.append(row);});li.append(list);$('rule-trace').append(li);
    });
  }
  function revealResult() {
    $('output-title').tabIndex=-1;
    $('output-title').focus({preventScroll:true});
    if(matchMedia('(max-width: 900px)').matches)$('output-title').scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  }
  function run() {
    try {lastRun={...engine.resolve(data,draft),record:clone(draft)};pending=false;renderResult();}
    catch(error){lastRun=null;pending=false;$('run-result').hidden=true;$('run-status').textContent=text('Cannot run these conditions: ','无法运行这些条件：')+error.message;}
  }
  function translate(next) {
    language=next==='zh'?'zh':'en';document.documentElement.lang=language==='zh'?'zh-CN':'en';
    document.title=text('Run a case — PersonaGuard','运行案例 — PersonaGuard');
    document.querySelectorAll('[data-en][data-zh]').forEach(node=>{node.textContent=node.dataset[language];});
    document.querySelectorAll('[data-lang]').forEach(node=>node.setAttribute('aria-pressed',String(node.dataset.lang===language)));
    document.querySelectorAll('[data-field]').forEach(node=>{for(const option of node.options)option.textContent=optionLabel(option.value,node.dataset.field);});
    for(const group of $('sample-select').children){group.label=group.dataset.group==='worked'?text('MER-PS worked cases','MER-PS 工作案例'):text('Author-coded external studies','作者编码的外部研究');for(const option of group.children){const item=data.samples.find(x=>x.record.id===option.value);option.textContent=caseCopy[item.record.id]?.summary_use[language] || item.record['label_'+language];}}
    if(sample){renderSample();renderResult();}
    if(startupError)$('startup-status').textContent=text('Local demo could not start. Reload the page or view the recorded results on the project page. ','本地运行未能启动。请刷新页面，或返回主页查看已有结果。')+startupError;
    try{localStorage.setItem('personaguard-language',language);}catch(_){}
  }
  document.querySelectorAll('[data-lang]').forEach(node=>node.addEventListener('click',()=>translate(node.dataset.lang)));
  try {
    if(!data||!engine)throw Error('Local rules or engine unavailable');
    // Refuse an inconsistent generated package before making the form usable.
    for(const item of data.samples){const result=engine.resolve(data,item.record).result;for(const key of ['audited_action','matched_rule_id'])if(result[key]!==item.baseline[key])throw Error('Case replay mismatch: '+item.record.id);}
    for(const groupName of ['worked','external']){const group=document.createElement('optgroup');group.dataset.group=groupName;for(const item of data.samples.filter(x=>x.group===groupName)){const option=document.createElement('option');option.value=item.record.id;group.append(option);}$('sample-select').append(group);}
    createFields();translate(language);selectSample(new URLSearchParams(location.search).get('case'));
    $('sample-select').addEventListener('change',()=>selectSample($('sample-select').value));
    $('run-form').addEventListener('submit',event=>{event.preventDefault();run();revealResult();});
    $('reset-case').addEventListener('click',()=>selectSample(sample.record.id));
    $('try-user-study').addEventListener('click',()=>{draft.evidence.evaluation_mode='IN_CONTEXT_USER_STUDY';draft.evidence.user_outcome='DIRECT_TASK_AND_EXPERIENCE';syncControls();run();revealResult();});
    $('download-result').addEventListener('click',()=>{
      if(pending||!lastRun)return;
      const record=lastRun.record,result=lastRun.result,changed=changes(record);
      const report={schema_version:'personaguard-browser-run-v1',protocol_version:data.protocol.protocol_version,created_at:new Date().toISOString(),
        hypothetical:!!changed.length,note:'Browser demonstration only. Modified conditions are assumptions, not new empirical findings or deployment permission.',
        source_sha256:data.source_sha256,source_case_id:sample.record.id,source_path:sample.source_path,
        original_result:sample.baseline,conditions:{deployment_stage:record.deployment_stage,route_capability:record.route_capability,evidence:record.evidence},
        changes:changed,result:{audited_action:result.audited_action,matched_rule_id:result.matched_rule_id,decisive_values:result.decisive_values},trace:lastRun.trace};
      const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)+'\n'],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='personaguard-'+sample.record.id+'-run.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
    $('startup-status').hidden=true;$('lab-layout').hidden=false;
  }catch(error){startupError=error.message;$('startup-status').textContent=text('Local demo could not start. Reload the page or view the recorded results on the project page. ','本地运行未能启动。请刷新页面，或返回主页查看已有结果。')+error.message;}
})();
