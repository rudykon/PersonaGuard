window.PERSONAGUARD_CASES = {
  "protocol_version": "2.0",
  "cases": [
    {
      "audited_action": "RETAIN_EVALUATED_COMPARATOR",
      "audited_decision_en": "Retain the evaluated comparator for this route",
      "audited_decision_zh": "为该路径保留已评估的比较方案",
      "decision_changed": true,
      "decisive_fields": [
        "evidence.comparator_increment"
      ],
      "decisive_values": {
        "evidence.comparator_increment": "NO_DEMONSTRATED_INCREMENT"
      },
      "evidence_basis_en": "Every evaluated sensor-containing point estimate was worse than video mean under participant-grouped nested evaluation.",
      "evidence_basis_zh": "在 participant-grouped 嵌套评估中，所有含传感器方案的点估计均差于 video mean。",
      "id": "optional_sensing",
      "label_en": "Optional EEG/fNIRS acquisition",
      "label_zh": "可选 EEG/fNIRS 获取路径",
      "matched_rule_id": "R2_COMPARATOR_INCREMENT",
      "matched_rule_kind": "constraint",
      "naive_action": "ADD_SENSORS_FOR_PERSONALIZATION",
      "proposed_use_en": "Add physiology to predict the same profile without a target-video joystick trace.",
      "proposed_use_zh": "增加生理信号，在没有目标视频 joystick 轨迹时预测同一画像。",
      "source_id": "merps_worked_case",
      "source_locator": "results/evidence_traceability.json#eeg_fnirs_increment"
    },
    {
      "audited_action": "WITHHOLD_RETENTION_AND_TRANSFER",
      "audited_decision_en": "Withhold durable retention and cross-context transfer",
      "audited_decision_zh": "不进行持久保留与跨情境迁移",
      "decision_changed": true,
      "decisive_fields": [
        "deployment_stage",
        "evidence.persistence",
        "evidence.reference_equity",
        "evidence.transfer"
      ],
      "decisive_values": {
        "deployment_stage": "RETENTION_TRANSFER",
        "evidence.persistence": "NOT_TESTED",
        "evidence.reference_equity": "NOT_ASSESSABLE",
        "evidence.transfer": "NOT_TESTED"
      },
      "evidence_basis_en": "No repeat session, cross-interface test, downstream utility study, or assessable reference-coverage analysis exists.",
      "evidence_basis_zh": "不存在重复 session、跨界面检验、下游效用研究或可评估的参考覆盖分析。",
      "id": "retention_transfer",
      "label_en": "Durable retention and transfer",
      "label_zh": "持久保留与迁移",
      "matched_rule_id": "R4_RETENTION_TRANSFER",
      "matched_rule_kind": "constraint",
      "naive_action": "RETAIN_AND_TRANSFER_PROFILE",
      "proposed_use_en": "Store the participant-indexed profile and reuse it in later sessions or interfaces.",
      "proposed_use_zh": "保存参与者画像并在后续 session 或界面中复用。",
      "source_id": "merps_worked_case",
      "source_locator": "results/evidence_traceability.json#retention_and_transfer"
    },
    {
      "audited_action": "RUN_PREREGISTERED_USER_STUDY",
      "audited_decision_en": "Run a preregistered in-context user study before deployment",
      "audited_decision_zh": "部署前开展预注册的情境内用户研究",
      "decision_changed": true,
      "decisive_fields": [
        "deployment_stage",
        "evidence.evaluation_mode",
        "evidence.user_outcome"
      ],
      "decisive_values": {
        "deployment_stage": "CONSEQUENTIAL_PERSONALIZATION",
        "evidence.evaluation_mode": "PROXY_ONLY",
        "evidence.user_outcome": "PROXIMAL_ONLY"
      },
      "evidence_basis_en": "The reference-proximal correction improves a trace endpoint, but no user benefit, meaningful-gain threshold, or acceptable harm rate was tested.",
      "evidence_basis_zh": "参考近端校正改善了轨迹终点，但尚未检验用户收益、最小有意义改善或可接受伤害比例。",
      "id": "signed_calibration",
      "label_en": "Signed behavioral calibration",
      "label_zh": "有符号行为校准",
      "matched_rule_id": "R3_CONSEQUENCE_MATCH",
      "matched_rule_kind": "constraint",
      "naive_action": "DEPLOY_CALIBRATION",
      "proposed_use_en": "Deploy a constant signed correction inferred from separate calibration videos.",
      "proposed_use_zh": "部署由独立校准视频推断的常量有符号校正。",
      "source_id": "merps_worked_case",
      "source_locator": "results/evidence_traceability.json#signed_proximal_correction"
    },
    {
      "audited_action": "BOUND_TO_EVIDENCE_SCOPE",
      "audited_decision_en": "Use the route only inside its evaluated evidence boundary",
      "audited_decision_zh": "仅在已评估证据边界内使用该路径",
      "decision_changed": true,
      "decisive_fields": [
        "evidence.interpretation_scope",
        "route_capability"
      ],
      "decisive_values": {
        "evidence.interpretation_scope": "PARTIAL",
        "route_capability": "AVAILABLE"
      },
      "evidence_basis_en": "The trace is computable, but its magnitude changes with interface, estimator, and shared reference; no repeat session exists.",
      "evidence_basis_zh": "轨迹可以计算，但其幅度随界面、估计器和共享参考而变化，且没有重复 session。",
      "id": "trace_interpretation",
      "label_en": "Trace-derived profile interpretation",
      "label_zh": "轨迹派生画像解释",
      "matched_rule_id": "R1_SCOPE_MATCH",
      "matched_rule_kind": "constraint",
      "naive_action": "RETAIN_AS_DURABLE_PERSON_PROFILE",
      "proposed_use_en": "Compute and interpret a post-trial temporal-deviation profile.",
      "proposed_use_zh": "计算并解释试次后的时间偏差画像。",
      "source_id": "merps_worked_case",
      "source_locator": "results/evidence_traceability.json#measurement_meaning"
    }
  ],
  "rules": [
    {
      "id": "R4_RETENTION_TRANSFER",
      "title_en": "Persistence, transfer, and reference coverage must match durable reuse",
      "title_zh": "持久性、迁移与参考覆盖必须匹配长期复用",
      "action": "WITHHOLD_RETENTION_AND_TRANSFER"
    },
    {
      "id": "R2_COMPARATOR_INCREMENT",
      "title_en": "Added acquisition or adaptation must beat its route-matched comparator",
      "title_zh": "新增获取或自适应必须优于路径匹配比较方案",
      "action": "RETAIN_EVALUATED_COMPARATOR"
    },
    {
      "id": "R3_CONSEQUENCE_MATCH",
      "title_en": "Evidence must match the consequence and be observed in use",
      "title_zh": "证据必须匹配后果并在真实使用中观察",
      "action": "RUN_PREREGISTERED_USER_STUDY"
    },
    {
      "id": "R1_SCOPE_MATCH",
      "title_en": "Interpretation and use cannot exceed the evaluated evidence scope",
      "title_zh": "解释与用途不得超出已评估证据范围",
      "action": "BOUND_TO_EVIDENCE_SCOPE"
    },
    {
      "id": "R5_BOUNDED_PASS",
      "title_en": "Matched direct evidence and reversible use can license bounded personalization",
      "title_zh": "匹配的直接证据与可逆使用可许可受限个性化",
      "action": "PROCEED_WITHIN_EVALUATED_BOUNDARY"
    }
  ]
};
