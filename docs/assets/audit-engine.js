/* Declarative browser counterpart of scripts/run_protocol_replay.py.
 * Reads predicates and vocabularies from generated public inputs, never labels
 * or expected actions from a sample. No network, storage, or DOM access. */
((root) => {
  'use strict';
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const fail = message => { throw new Error(message); };
  function field(record, path) {
    return path.split('.').reduce((value, key) => {
      if (!object(value) || !own(value, key)) fail(`Missing field: ${path}`);
      return value[key];
    }, record);
  }
  function inspect(predicate, record, legalFields) {
    if (!object(predicate)) fail('Invalid predicate');
    const groups = ['all', 'any'].filter(key => own(predicate, key));
    if (groups.length) {
      if (groups.length !== 1 || Object.keys(predicate).length !== 1) fail('Ambiguous predicate');
      const op = groups[0], children = predicate[op];
      if (!Array.isArray(children) || !children.length) fail('Empty predicate group');
      const checked = children.map(child => inspect(child, record, legalFields));
      return {matches: op === 'all' ? checked.every(x => x.matches) : checked.some(x => x.matches),
        fields: [...new Set(checked.flatMap(x => x.fields))].sort()};
    }
    const path = predicate.field;
    if (typeof path !== 'string' || !legalFields.has(path)) fail('Unknown predicate field');
    const ops = ['eq', 'in', 'not_in'].filter(key => own(predicate, key));
    if (ops.length !== 1 || Object.keys(predicate).length !== 2) fail('Invalid predicate operator');
    const observed = field(record, path), op = ops[0], expected = predicate[op];
    if (op !== 'eq' && (!Array.isArray(expected) || !expected.length)) fail('Empty predicate values');
    const matches = op === 'eq' ? observed === expected : op === 'in' ? expected.includes(observed) : !expected.includes(observed);
    return {matches, fields: [path]};
  }
  function resolve(data, record) {
    if (!object(data) || !object(data.protocol)) fail('Missing protocol');
    const protocol = data.protocol;
    if (protocol.schema_version !== 'route-audit-protocol-v2') fail('Unsupported protocol schema');
    if (!object(record)) fail('Record must be an object');
    const required = protocol.required_route_fields, vocabulary = protocol.evidence_fields;
    if (!Array.isArray(required) || !required.length || !object(vocabulary)) fail('Invalid schema');
    for (const key of required) if (!own(record, key)) fail(`Missing field: ${key}`);
    for (const key of data.forbidden_case_fields) if (own(record, key)) fail(`Record encodes an answer: ${key}`);
    if (!data.stages.includes(record.deployment_stage)) fail('Unknown deployment stage');
    if (!data.capabilities.includes(record.route_capability)) fail('Unknown route capability');
    if (!object(record.evidence) || Object.keys(record.evidence).sort().join('|') !== Object.keys(vocabulary).sort().join('|')) fail('Evidence fields differ from protocol');
    for (const [key, values] of Object.entries(vocabulary)) {
      if (!Array.isArray(values) || !values.includes(record.evidence[key])) fail(`Invalid evidence: ${key}`);
    }
    const legal = new Set([...required, ...Object.keys(vocabulary).map(key => `evidence.${key}`)]);
    if (!Array.isArray(protocol.rules) || !protocol.rules.length) fail('Missing rules');
    const rules = [...protocol.rules].sort((a, b) => a.priority - b.priority);
    const ids = new Set(), priorities = new Set();
    const trace = rules.map(rule => {
      if (ids.has(rule.id) || priorities.has(rule.priority) || !Number.isFinite(rule.priority)) fail('Invalid rule identity or priority');
      ids.add(rule.id); priorities.add(rule.priority);
      if (!['constraint', 'permission'].includes(rule.kind) || !own(protocol.action_labels, rule.action)) fail('Invalid rule action');
      const check = inspect(rule.predicate, record, legal);
      return {id: rule.id, priority: rule.priority, kind: rule.kind, action: rule.action,
        matches: check.matches, fields: check.fields,
        values: Object.fromEntries(check.fields.map(path => [path, field(record, path)]))};
    });
    const winner = trace.find(item => item.matches);
    const action = winner ? winner.action : protocol.resolver.fallback_action;
    const labels = protocol.action_labels[action];
    if (!labels || typeof labels.en !== 'string' || typeof labels.zh !== 'string') fail('Missing action label');
    trace.forEach(item => { item.selected = item === winner; });
    const result = {};
    for (const key of ['id', 'source_id', 'label_en', 'label_zh', 'proposed_use_en', 'proposed_use_zh', 'evidence_basis_en', 'evidence_basis_zh', 'source_locator', 'naive_action']) result[key] = record[key];
    Object.assign(result, {audited_action: action, audited_decision_en: labels.en, audited_decision_zh: labels.zh,
      matched_rule_id: winner ? winner.id : 'FALLBACK', matched_rule_kind: winner ? winner.kind : 'evidence_request',
      decisive_fields: winner ? winner.fields : [], decisive_values: winner ? winner.values : {}, decision_changed: record.naive_action !== action});
    return {result, trace};
  }
  const api = Object.freeze({resolve});
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PERSONAGUARD_AUDIT = api;
})(globalThis);
