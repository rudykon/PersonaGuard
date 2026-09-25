window.PERSONAGUARD_AUDIT_DATA = {
  "schema_version": "personaguard-browser-inputs-v1",
  "protocol": {
    "schema_version": "route-audit-protocol-v2",
    "protocol_version": "2.0",
    "unit_of_audit": "A candidate personalization route defined by its proposed use, deployment stage, comparator, evidence boundary, consequence, and retention horizon.",
    "design_status": "Five normative design commitments synthesized from validity, HCI personalization, documentation, and algorithmic-audit literature. The sources motivate the commitments; they do not logically prove that the rules are complete or universally correct.",
    "required_route_fields": [
      "id",
      "source_id",
      "label_en",
      "label_zh",
      "proposed_use_en",
      "proposed_use_zh",
      "deployment_stage",
      "route_capability",
      "evidence",
      "evidence_basis_en",
      "evidence_basis_zh",
      "source_locator",
      "naive_action"
    ],
    "evidence_fields": {
      "interpretation_scope": [
        "MATCHED",
        "PARTIAL",
        "NOT_TESTED",
        "NOT_APPLICABLE"
      ],
      "comparator_increment": [
        "SUPPORTED",
        "PARTIAL",
        "NO_DEMONSTRATED_INCREMENT",
        "NOT_TESTED",
        "NOT_APPLICABLE"
      ],
      "evaluation_mode": [
        "IN_CONTEXT_USER_STUDY",
        "TECHNICAL_AND_EMPIRICAL_EVALUATION",
        "CASE_STUDY_IN_USE",
        "LONGITUDINAL_FIELD_USE",
        "PROXY_ONLY",
        "VIGNETTE_ONLY",
        "NOT_TESTED",
        "NOT_APPLICABLE"
      ],
      "user_outcome": [
        "DIRECT_TASK_AND_EXPERIENCE",
        "DIRECT_USER_EXPERIENCE",
        "PROXIMAL_ONLY",
        "PREFERENCE_ONLY",
        "NOT_TESTED",
        "NOT_APPLICABLE"
      ],
      "user_control": [
        "SUPPORTED",
        "PARTIAL",
        "NOT_TESTED",
        "NOT_REPORTED",
        "NOT_APPLICABLE"
      ],
      "safety_reversibility": [
        "USER_CONTROLLED",
        "REVERSIBLE",
        "PARTIAL",
        "NOT_TESTED",
        "NOT_APPLICABLE"
      ],
      "persistence": [
        "SUPPORTED",
        "PARTIAL",
        "NOT_TESTED",
        "NOT_REQUIRED"
      ],
      "transfer": [
        "SUPPORTED",
        "PARTIAL",
        "NOT_TESTED",
        "NOT_REQUIRED"
      ],
      "reference_equity": [
        "SUPPORTED",
        "PARTIAL",
        "NOT_ASSESSABLE",
        "NOT_REQUIRED"
      ]
    },
    "action_labels": {
      "BOUND_TO_EVIDENCE_SCOPE": {
        "en": "Use the route only inside its evaluated evidence boundary",
        "zh": "仅在已评估证据边界内使用该路径"
      },
      "RETAIN_EVALUATED_COMPARATOR": {
        "en": "Retain the evaluated comparator for this route",
        "zh": "为该路径保留已评估的比较方案"
      },
      "RUN_PREREGISTERED_USER_STUDY": {
        "en": "Run a preregistered in-context user study before deployment",
        "zh": "部署前开展预注册的情境内用户研究"
      },
      "WITHHOLD_RETENTION_AND_TRANSFER": {
        "en": "Withhold durable retention and cross-context transfer",
        "zh": "不进行持久保留与跨情境迁移"
      },
      "PROCEED_WITHIN_EVALUATED_BOUNDARY": {
        "en": "Proceed with personalization inside the evaluated boundary",
        "zh": "在已评估边界内实施个性化"
      },
      "REQUIRE_ROUTE_SPECIFIC_EVIDENCE": {
        "en": "Collect the route-specific evidence named by the unresolved record",
        "zh": "补充未决记录所指明的路径特定证据"
      }
    },
    "rules": [
      {
        "id": "R4_RETENTION_TRANSFER",
        "priority": 10,
        "kind": "constraint",
        "title_en": "Persistence, transfer, and reference coverage must match durable reuse",
        "title_zh": "持久性、迁移与参考覆盖必须匹配长期复用",
        "risk_blocked_en": "A technically computable profile is silently converted into a durable or portable person property.",
        "predicate": {
          "all": [
            {
              "field": "deployment_stage",
              "eq": "RETENTION_TRANSFER"
            },
            {
              "any": [
                {
                  "field": "evidence.persistence",
                  "in": [
                    "PARTIAL",
                    "NOT_TESTED"
                  ]
                },
                {
                  "field": "evidence.transfer",
                  "in": [
                    "PARTIAL",
                    "NOT_TESTED"
                  ]
                },
                {
                  "field": "evidence.reference_equity",
                  "in": [
                    "PARTIAL",
                    "NOT_ASSESSABLE"
                  ]
                }
              ]
            }
          ]
        },
        "action": "WITHHOLD_RETENTION_AND_TRANSFER",
        "derivation_sources": [
          "mitchell2019modelcards",
          "gebru2021datasheets",
          "raji2020accountability",
          "raij2011privacy"
        ]
      },
      {
        "id": "R2_COMPARATOR_INCREMENT",
        "priority": 20,
        "kind": "constraint",
        "title_en": "Added acquisition or adaptation must beat its route-matched comparator",
        "title_zh": "新增获取或自适应必须优于路径匹配比较方案",
        "risk_blocked_en": "Complexity, burden, or automatic change is accepted merely because a personalized model can be fitted.",
        "predicate": {
          "field": "evidence.comparator_increment",
          "eq": "NO_DEMONSTRATED_INCREMENT"
        },
        "action": "RETAIN_EVALUATED_COMPARATOR",
        "derivation_sources": [
          "findlater2004menus",
          "todi2021modelbased",
          "raji2020accountability"
        ]
      },
      {
        "id": "R3_CONSEQUENCE_MATCH",
        "priority": 30,
        "kind": "constraint",
        "title_en": "Evidence must match the consequence and be observed in use",
        "title_zh": "证据必须匹配后果并在真实使用中观察",
        "risk_blocked_en": "A proxy correction, preference, or vignette response is relabeled as demonstrated user benefit.",
        "predicate": {
          "all": [
            {
              "field": "deployment_stage",
              "in": [
                "REVERSIBLE_PERSONALIZATION",
                "CONSEQUENTIAL_PERSONALIZATION"
              ]
            },
            {
              "any": [
                {
                  "field": "evidence.evaluation_mode",
                  "in": [
                    "PROXY_ONLY",
                    "VIGNETTE_ONLY",
                    "NOT_TESTED"
                  ]
                },
                {
                  "field": "evidence.user_outcome",
                  "in": [
                    "PROXIMAL_ONLY",
                    "PREFERENCE_ONLY",
                    "NOT_TESTED"
                  ]
                }
              ]
            }
          ]
        },
        "action": "RUN_PREREGISTERED_USER_STUDY",
        "derivation_sources": [
          "amershi2019guidelines",
          "madaio2020checklists",
          "alves2026interaction"
        ]
      },
      {
        "id": "R1_SCOPE_MATCH",
        "priority": 40,
        "kind": "constraint",
        "title_en": "Interpretation and use cannot exceed the evaluated evidence scope",
        "title_zh": "解释与用途不得超出已评估证据范围",
        "risk_blocked_en": "A context-dependent score is promoted to a stable person property without the required evidence.",
        "predicate": {
          "all": [
            {
              "field": "route_capability",
              "eq": "AVAILABLE"
            },
            {
              "field": "evidence.interpretation_scope",
              "in": [
                "PARTIAL",
                "NOT_TESTED"
              ]
            }
          ]
        },
        "action": "BOUND_TO_EVIDENCE_SCOPE",
        "derivation_sources": [
          "messick1995validity",
          "kane2013validating",
          "jacobs2021measurement"
        ]
      },
      {
        "id": "R5_BOUNDED_PASS",
        "priority": 90,
        "kind": "permission",
        "title_en": "Matched direct evidence and reversible use can license bounded personalization",
        "title_zh": "匹配的直接证据与可逆使用可许可受限个性化",
        "risk_blocked_en": "Without a positive rule, the audit becomes a missing-data checklist that can constrain routes but never authorize a supported one.",
        "predicate": {
          "all": [
            {
              "field": "deployment_stage",
              "eq": "REVERSIBLE_PERSONALIZATION"
            },
            {
              "field": "route_capability",
              "eq": "AVAILABLE"
            },
            {
              "field": "evidence.interpretation_scope",
              "eq": "MATCHED"
            },
            {
              "field": "evidence.comparator_increment",
              "in": [
                "SUPPORTED",
                "PARTIAL",
                "NOT_APPLICABLE"
              ]
            },
            {
              "field": "evidence.evaluation_mode",
              "in": [
                "IN_CONTEXT_USER_STUDY",
                "TECHNICAL_AND_EMPIRICAL_EVALUATION",
                "CASE_STUDY_IN_USE",
                "LONGITUDINAL_FIELD_USE"
              ]
            },
            {
              "field": "evidence.user_outcome",
              "in": [
                "DIRECT_TASK_AND_EXPERIENCE",
                "DIRECT_USER_EXPERIENCE"
              ]
            },
            {
              "any": [
                {
                  "field": "evidence.user_control",
                  "eq": "SUPPORTED"
                },
                {
                  "field": "evidence.safety_reversibility",
                  "in": [
                    "USER_CONTROLLED",
                    "REVERSIBLE"
                  ]
                }
              ]
            }
          ]
        },
        "action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        "derivation_sources": [
          "gajos2010supple",
          "reinecke2011cultural",
          "johnson2013calibration",
          "findlater2004menus"
        ]
      }
    ],
    "resolver": {
      "policy": "Evaluate rules by ascending priority. The first matching constraint wins; a bounded-pass permission is returned only when no higher-priority constraint matches. No match returns a named evidence request, never silent approval.",
      "precedence_rationale": "Lifecycle obligations precede local performance; a failed route-matched comparator precedes consequence and scope; consequence mismatch precedes scope; positive permission is evaluated last. This harm-first order returns the highest-consequence unresolved obligation rather than treating priorities as empirical weights.",
      "fallback_action": "REQUIRE_ROUTE_SPECIFIC_EVIDENCE",
      "ablation_semantics": "Removing a rule removes its constraint or permission. An affected route becomes unresolved unless another declared rule independently matches; ablation does not assume automatic deployment."
    },
    "machine_check_scope": {
      "checked": [
        "route and rule schema",
        "declared field vocabularies",
        "predicate evaluation and priority resolution",
        "source locators and content hashes where local artifacts exist",
        "order invariance, malformed-record rejection, locality, and generated-file propagation",
        "rule-wise action and obligation ablation",
        "positive-permission boundary and declared precedence probes"
      ],
      "not_checked": [
        "truth or completeness of literature coding",
        "substantive validity of a route status or rule",
        "human analyst agreement, completion time, or usability",
        "fairness, user benefit, or safety outside each source's evaluated boundary"
      ]
    }
  },
  "stages": [
    "CONSEQUENTIAL_PERSONALIZATION",
    "OFFLINE_ESTIMATION",
    "OPTIONAL_ACQUISITION",
    "RETENTION_TRANSFER",
    "REVERSIBLE_PERSONALIZATION"
  ],
  "capabilities": [
    "AVAILABLE",
    "NOT_AVAILABLE",
    "PARTIAL"
  ],
  "forbidden_case_fields": [
    "audited_action",
    "expected_action",
    "rule_id"
  ],
  "source_sha256": {
    "configs/protocol/protocol_rules.json": "c9410b79e254b0b087b8bc766668ac90dd842ec53b8fafe644baff5d057ffe57",
    "configs/protocol/protocol_replay_cases.json": "48da358ee19744d8aff321c7b6df988379d7d80139a080edb77e6e8f72ec586d",
    "configs/protocol/external_reuse_cases.json": "30165db6a88e7c4e627d7e5d5fcd1cfb5481a3eec9ab90fed753e44b0d64f174"
  },
  "samples": [
    {
      "record": {
        "id": "trace_interpretation",
        "source_id": "merps_worked_case",
        "label_en": "Trace-derived profile interpretation",
        "label_zh": "轨迹派生画像解释",
        "proposed_use_en": "Compute and interpret a post-trial temporal-deviation profile.",
        "proposed_use_zh": "计算并解释试次后的时间偏差画像。",
        "deployment_stage": "OFFLINE_ESTIMATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "PARTIAL",
          "comparator_increment": "NOT_APPLICABLE",
          "evaluation_mode": "PROXY_ONLY",
          "user_outcome": "NOT_APPLICABLE",
          "user_control": "NOT_APPLICABLE",
          "safety_reversibility": "NOT_APPLICABLE",
          "persistence": "NOT_TESTED",
          "transfer": "NOT_TESTED",
          "reference_equity": "NOT_ASSESSABLE"
        },
        "evidence_basis_en": "The trace is computable, but its magnitude changes with interface, estimator, and shared reference; no repeat session exists.",
        "evidence_basis_zh": "轨迹可以计算，但其幅度随界面、估计器和共享参考而变化，且没有重复 session。",
        "source_locator": "results/evidence_traceability.json#measurement_meaning",
        "naive_action": "RETAIN_AS_DURABLE_PERSON_PROFILE"
      },
      "group": "worked",
      "source_path": "configs/protocol/protocol_replay_cases.json",
      "baseline": {
        "id": "trace_interpretation",
        "source_id": "merps_worked_case",
        "label_en": "Trace-derived profile interpretation",
        "label_zh": "轨迹派生画像解释",
        "proposed_use_en": "Compute and interpret a post-trial temporal-deviation profile.",
        "proposed_use_zh": "计算并解释试次后的时间偏差画像。",
        "evidence_basis_en": "The trace is computable, but its magnitude changes with interface, estimator, and shared reference; no repeat session exists.",
        "evidence_basis_zh": "轨迹可以计算，但其幅度随界面、估计器和共享参考而变化，且没有重复 session。",
        "source_locator": "results/evidence_traceability.json#measurement_meaning",
        "naive_action": "RETAIN_AS_DURABLE_PERSON_PROFILE",
        "audited_action": "BOUND_TO_EVIDENCE_SCOPE",
        "audited_decision_en": "Use the route only inside its evaluated evidence boundary",
        "audited_decision_zh": "仅在已评估证据边界内使用该路径",
        "matched_rule_id": "R1_SCOPE_MATCH",
        "matched_rule_kind": "constraint",
        "decisive_fields": [
          "evidence.interpretation_scope",
          "route_capability"
        ],
        "decisive_values": {
          "evidence.interpretation_scope": "PARTIAL",
          "route_capability": "AVAILABLE"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "optional_sensing",
        "source_id": "merps_worked_case",
        "label_en": "Optional EEG/fNIRS acquisition",
        "label_zh": "可选 EEG/fNIRS 获取路径",
        "proposed_use_en": "Add physiology to predict the same profile without a target-video joystick trace.",
        "proposed_use_zh": "增加生理信号，在没有目标视频 joystick 轨迹时预测同一画像。",
        "deployment_stage": "OPTIONAL_ACQUISITION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "PARTIAL",
          "comparator_increment": "NO_DEMONSTRATED_INCREMENT",
          "evaluation_mode": "PROXY_ONLY",
          "user_outcome": "PROXIMAL_ONLY",
          "user_control": "NOT_APPLICABLE",
          "safety_reversibility": "PARTIAL",
          "persistence": "NOT_TESTED",
          "transfer": "NOT_TESTED",
          "reference_equity": "NOT_ASSESSABLE"
        },
        "evidence_basis_en": "Every evaluated sensor-containing point estimate was worse than video mean under participant-grouped nested evaluation.",
        "evidence_basis_zh": "在 participant-grouped 嵌套评估中，所有含传感器方案的点估计均差于 video mean。",
        "source_locator": "results/evidence_traceability.json#eeg_fnirs_increment",
        "naive_action": "ADD_SENSORS_FOR_PERSONALIZATION"
      },
      "group": "worked",
      "source_path": "configs/protocol/protocol_replay_cases.json",
      "baseline": {
        "id": "optional_sensing",
        "source_id": "merps_worked_case",
        "label_en": "Optional EEG/fNIRS acquisition",
        "label_zh": "可选 EEG/fNIRS 获取路径",
        "proposed_use_en": "Add physiology to predict the same profile without a target-video joystick trace.",
        "proposed_use_zh": "增加生理信号，在没有目标视频 joystick 轨迹时预测同一画像。",
        "evidence_basis_en": "Every evaluated sensor-containing point estimate was worse than video mean under participant-grouped nested evaluation.",
        "evidence_basis_zh": "在 participant-grouped 嵌套评估中，所有含传感器方案的点估计均差于 video mean。",
        "source_locator": "results/evidence_traceability.json#eeg_fnirs_increment",
        "naive_action": "ADD_SENSORS_FOR_PERSONALIZATION",
        "audited_action": "RETAIN_EVALUATED_COMPARATOR",
        "audited_decision_en": "Retain the evaluated comparator for this route",
        "audited_decision_zh": "为该路径保留已评估的比较方案",
        "matched_rule_id": "R2_COMPARATOR_INCREMENT",
        "matched_rule_kind": "constraint",
        "decisive_fields": [
          "evidence.comparator_increment"
        ],
        "decisive_values": {
          "evidence.comparator_increment": "NO_DEMONSTRATED_INCREMENT"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "signed_calibration",
        "source_id": "merps_worked_case",
        "label_en": "Signed behavioral calibration",
        "label_zh": "有符号行为校准",
        "proposed_use_en": "Deploy a constant signed correction inferred from separate calibration videos.",
        "proposed_use_zh": "部署由独立校准视频推断的常量有符号校正。",
        "deployment_stage": "CONSEQUENTIAL_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "PARTIAL",
          "comparator_increment": "SUPPORTED",
          "evaluation_mode": "PROXY_ONLY",
          "user_outcome": "PROXIMAL_ONLY",
          "user_control": "NOT_TESTED",
          "safety_reversibility": "REVERSIBLE",
          "persistence": "NOT_TESTED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "NOT_ASSESSABLE"
        },
        "evidence_basis_en": "The reference-proximal correction improves a trace endpoint, but no user benefit, meaningful-gain threshold, or acceptable harm rate was tested.",
        "evidence_basis_zh": "参考近端校正改善了轨迹终点，但尚未检验用户收益、最小有意义改善或可接受伤害比例。",
        "source_locator": "results/evidence_traceability.json#signed_proximal_correction",
        "naive_action": "DEPLOY_CALIBRATION"
      },
      "group": "worked",
      "source_path": "configs/protocol/protocol_replay_cases.json",
      "baseline": {
        "id": "signed_calibration",
        "source_id": "merps_worked_case",
        "label_en": "Signed behavioral calibration",
        "label_zh": "有符号行为校准",
        "proposed_use_en": "Deploy a constant signed correction inferred from separate calibration videos.",
        "proposed_use_zh": "部署由独立校准视频推断的常量有符号校正。",
        "evidence_basis_en": "The reference-proximal correction improves a trace endpoint, but no user benefit, meaningful-gain threshold, or acceptable harm rate was tested.",
        "evidence_basis_zh": "参考近端校正改善了轨迹终点，但尚未检验用户收益、最小有意义改善或可接受伤害比例。",
        "source_locator": "results/evidence_traceability.json#signed_proximal_correction",
        "naive_action": "DEPLOY_CALIBRATION",
        "audited_action": "RUN_PREREGISTERED_USER_STUDY",
        "audited_decision_en": "Run a preregistered in-context user study before deployment",
        "audited_decision_zh": "部署前开展预注册的情境内用户研究",
        "matched_rule_id": "R3_CONSEQUENCE_MATCH",
        "matched_rule_kind": "constraint",
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
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "retention_transfer",
        "source_id": "merps_worked_case",
        "label_en": "Durable retention and transfer",
        "label_zh": "持久保留与迁移",
        "proposed_use_en": "Store the participant-indexed profile and reuse it in later sessions or interfaces.",
        "proposed_use_zh": "保存参与者画像并在后续 session 或界面中复用。",
        "deployment_stage": "RETENTION_TRANSFER",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "PARTIAL",
          "comparator_increment": "NOT_APPLICABLE",
          "evaluation_mode": "NOT_TESTED",
          "user_outcome": "NOT_TESTED",
          "user_control": "NOT_TESTED",
          "safety_reversibility": "PARTIAL",
          "persistence": "NOT_TESTED",
          "transfer": "NOT_TESTED",
          "reference_equity": "NOT_ASSESSABLE"
        },
        "evidence_basis_en": "No repeat session, cross-interface test, downstream utility study, or assessable reference-coverage analysis exists.",
        "evidence_basis_zh": "不存在重复 session、跨界面检验、下游效用研究或可评估的参考覆盖分析。",
        "source_locator": "results/evidence_traceability.json#retention_and_transfer",
        "naive_action": "RETAIN_AND_TRANSFER_PROFILE"
      },
      "group": "worked",
      "source_path": "configs/protocol/protocol_replay_cases.json",
      "baseline": {
        "id": "retention_transfer",
        "source_id": "merps_worked_case",
        "label_en": "Durable retention and transfer",
        "label_zh": "持久保留与迁移",
        "proposed_use_en": "Store the participant-indexed profile and reuse it in later sessions or interfaces.",
        "proposed_use_zh": "保存参与者画像并在后续 session 或界面中复用。",
        "evidence_basis_en": "No repeat session, cross-interface test, downstream utility study, or assessable reference-coverage analysis exists.",
        "evidence_basis_zh": "不存在重复 session、跨界面检验、下游效用研究或可评估的参考覆盖分析。",
        "source_locator": "results/evidence_traceability.json#retention_and_transfer",
        "naive_action": "RETAIN_AND_TRANSFER_PROFILE",
        "audited_action": "WITHHOLD_RETENTION_AND_TRANSFER",
        "audited_decision_en": "Withhold durable retention and cross-context transfer",
        "audited_decision_zh": "不进行持久保留与跨情境迁移",
        "matched_rule_id": "R4_RETENTION_TRANSFER",
        "matched_rule_kind": "constraint",
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
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "findlater_adaptive_menu",
        "source_id": "findlater2004menus",
        "label_en": "System-adaptive split menu",
        "label_zh": "系统自适应 split menu",
        "proposed_use_en": "Use frequency-and-recency adaptation instead of the evaluated static split menu.",
        "proposed_use_zh": "以频率和近期使用驱动的自适应替代已评估静态 split menu。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "NO_DEMONSTRATED_INCREMENT",
          "evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "user_control": "NOT_REPORTED",
          "safety_reversibility": "REVERSIBLE",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "NOT_REQUIRED"
        },
        "evidence_basis_en": "The adaptive menu was slower than the static comparator; the study therefore supplies route-specific negative evidence rather than missing data.",
        "evidence_basis_zh": "自适应 menu 慢于静态比较方案，因此这是路径特定的负证据，而非缺失数据。",
        "source_locator": "doi:10.1145/985692.985704#abstract-and-conclusion",
        "naive_action": "DEPLOY_SYSTEM_ADAPTATION"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "findlater_adaptive_menu",
        "source_id": "findlater2004menus",
        "label_en": "System-adaptive split menu",
        "label_zh": "系统自适应 split menu",
        "proposed_use_en": "Use frequency-and-recency adaptation instead of the evaluated static split menu.",
        "proposed_use_zh": "以频率和近期使用驱动的自适应替代已评估静态 split menu。",
        "evidence_basis_en": "The adaptive menu was slower than the static comparator; the study therefore supplies route-specific negative evidence rather than missing data.",
        "evidence_basis_zh": "自适应 menu 慢于静态比较方案，因此这是路径特定的负证据，而非缺失数据。",
        "source_locator": "doi:10.1145/985692.985704#abstract-and-conclusion",
        "naive_action": "DEPLOY_SYSTEM_ADAPTATION",
        "audited_action": "RETAIN_EVALUATED_COMPARATOR",
        "audited_decision_en": "Retain the evaluated comparator for this route",
        "audited_decision_zh": "为该路径保留已评估的比较方案",
        "matched_rule_id": "R2_COMPARATOR_INCREMENT",
        "matched_rule_kind": "constraint",
        "decisive_fields": [
          "evidence.comparator_increment"
        ],
        "decisive_values": {
          "evidence.comparator_increment": "NO_DEMONSTRATED_INCREMENT"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "findlater_adaptable_menu",
        "source_id": "findlater2004menus",
        "label_en": "User-adaptable split menu",
        "label_zh": "用户可调 split menu",
        "proposed_use_en": "Offer user-controlled split-menu customization in the evaluated menu task.",
        "proposed_use_zh": "在已评估 menu 任务中提供用户控制的 split-menu 定制。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "PARTIAL",
          "evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "user_control": "SUPPORTED",
          "safety_reversibility": "USER_CONTROLLED",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "NOT_REQUIRED"
        },
        "evidence_basis_en": "Participants could customize effectively in the tested task, adaptable menus beat adaptive menus in some orders, and most participants preferred the adaptable condition.",
        "evidence_basis_zh": "参与者可在测试任务中有效定制；可调 menu 在部分顺序中优于自适应 menu，且多数参与者偏好可调条件。",
        "source_locator": "doi:10.1145/985692.985704#abstract-and-conclusion",
        "naive_action": "REJECT_PERSONALIZATION_WHEN_ANY_COMPARISON_IS_MIXED"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "findlater_adaptable_menu",
        "source_id": "findlater2004menus",
        "label_en": "User-adaptable split menu",
        "label_zh": "用户可调 split menu",
        "proposed_use_en": "Offer user-controlled split-menu customization in the evaluated menu task.",
        "proposed_use_zh": "在已评估 menu 任务中提供用户控制的 split-menu 定制。",
        "evidence_basis_en": "Participants could customize effectively in the tested task, adaptable menus beat adaptive menus in some orders, and most participants preferred the adaptable condition.",
        "evidence_basis_zh": "参与者可在测试任务中有效定制；可调 menu 在部分顺序中优于自适应 menu，且多数参与者偏好可调条件。",
        "source_locator": "doi:10.1145/985692.985704#abstract-and-conclusion",
        "naive_action": "REJECT_PERSONALIZATION_WHEN_ANY_COMPARISON_IS_MIXED",
        "audited_action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        "audited_decision_en": "Proceed with personalization inside the evaluated boundary",
        "audited_decision_zh": "在已评估边界内实施个性化",
        "matched_rule_id": "R5_BOUNDED_PASS",
        "matched_rule_kind": "permission",
        "decisive_fields": [
          "deployment_stage",
          "evidence.comparator_increment",
          "evidence.evaluation_mode",
          "evidence.interpretation_scope",
          "evidence.safety_reversibility",
          "evidence.user_control",
          "evidence.user_outcome",
          "route_capability"
        ],
        "decisive_values": {
          "deployment_stage": "REVERSIBLE_PERSONALIZATION",
          "evidence.comparator_increment": "PARTIAL",
          "evidence.evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "evidence.interpretation_scope": "MATCHED",
          "evidence.safety_reversibility": "USER_CONTROLLED",
          "evidence.user_control": "SUPPORTED",
          "evidence.user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "route_capability": "AVAILABLE"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "supple_motor_access",
        "source_id": "gajos2010supple",
        "label_en": "Ability-based generated interfaces",
        "label_zh": "基于能力的生成界面",
        "proposed_use_en": "Generate reversible interfaces for users whose motor abilities are poorly served by manufacturer defaults.",
        "proposed_use_zh": "为不适合厂商默认界面的运动障碍用户生成可逆界面。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "SUPPORTED",
          "evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "user_control": "PARTIAL",
          "safety_reversibility": "REVERSIBLE",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "PARTIAL"
        },
        "evidence_basis_en": "Compared with manufacturer defaults, the evaluated interfaces improved speed, accuracy, and satisfaction for the target population.",
        "evidence_basis_zh": "相对厂商默认界面，已评估界面改善了目标人群的速度、准确性与满意度。",
        "source_locator": "doi:10.1016/j.artint.2010.05.005#abstract",
        "naive_action": "REJECT_PERSONALIZATION_UNTIL_ALL_TRANSFER_EVIDENCE_EXISTS"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "supple_motor_access",
        "source_id": "gajos2010supple",
        "label_en": "Ability-based generated interfaces",
        "label_zh": "基于能力的生成界面",
        "proposed_use_en": "Generate reversible interfaces for users whose motor abilities are poorly served by manufacturer defaults.",
        "proposed_use_zh": "为不适合厂商默认界面的运动障碍用户生成可逆界面。",
        "evidence_basis_en": "Compared with manufacturer defaults, the evaluated interfaces improved speed, accuracy, and satisfaction for the target population.",
        "evidence_basis_zh": "相对厂商默认界面，已评估界面改善了目标人群的速度、准确性与满意度。",
        "source_locator": "doi:10.1016/j.artint.2010.05.005#abstract",
        "naive_action": "REJECT_PERSONALIZATION_UNTIL_ALL_TRANSFER_EVIDENCE_EXISTS",
        "audited_action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        "audited_decision_en": "Proceed with personalization inside the evaluated boundary",
        "audited_decision_zh": "在已评估边界内实施个性化",
        "matched_rule_id": "R5_BOUNDED_PASS",
        "matched_rule_kind": "permission",
        "decisive_fields": [
          "deployment_stage",
          "evidence.comparator_increment",
          "evidence.evaluation_mode",
          "evidence.interpretation_scope",
          "evidence.safety_reversibility",
          "evidence.user_control",
          "evidence.user_outcome",
          "route_capability"
        ],
        "decisive_values": {
          "deployment_stage": "REVERSIBLE_PERSONALIZATION",
          "evidence.comparator_increment": "SUPPORTED",
          "evidence.evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "evidence.interpretation_scope": "MATCHED",
          "evidence.safety_reversibility": "REVERSIBLE",
          "evidence.user_control": "PARTIAL",
          "evidence.user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "route_capability": "AVAILABLE"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "culturally_adaptive_website",
        "source_id": "reinecke2011cultural",
        "label_en": "Culturally adaptive website",
        "label_zh": "文化自适应网站",
        "proposed_use_en": "Use the evaluated cultural adaptation for the tested website tasks and populations.",
        "proposed_use_zh": "在测试网站任务与人群中使用已评估文化自适应。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "SUPPORTED",
          "evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "user_control": "NOT_REPORTED",
          "safety_reversibility": "REVERSIBLE",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "PARTIAL"
        },
        "evidence_basis_en": "The adapted condition improved observed task performance and subjective experience against the non-adapted website in the evaluated sample.",
        "evidence_basis_zh": "在已评估样本中，自适应条件相对非自适应网站改善了任务表现与主观体验。",
        "source_locator": "doi:10.1145/1970378.1970382#abstract",
        "naive_action": "REJECT_PERSONALIZATION_BECAUSE_PROFILES_ARE_NEVER_PERMITTED"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "culturally_adaptive_website",
        "source_id": "reinecke2011cultural",
        "label_en": "Culturally adaptive website",
        "label_zh": "文化自适应网站",
        "proposed_use_en": "Use the evaluated cultural adaptation for the tested website tasks and populations.",
        "proposed_use_zh": "在测试网站任务与人群中使用已评估文化自适应。",
        "evidence_basis_en": "The adapted condition improved observed task performance and subjective experience against the non-adapted website in the evaluated sample.",
        "evidence_basis_zh": "在已评估样本中，自适应条件相对非自适应网站改善了任务表现与主观体验。",
        "source_locator": "doi:10.1145/1970378.1970382#abstract",
        "naive_action": "REJECT_PERSONALIZATION_BECAUSE_PROFILES_ARE_NEVER_PERMITTED",
        "audited_action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        "audited_decision_en": "Proceed with personalization inside the evaluated boundary",
        "audited_decision_zh": "在已评估边界内实施个性化",
        "matched_rule_id": "R5_BOUNDED_PASS",
        "matched_rule_kind": "permission",
        "decisive_fields": [
          "deployment_stage",
          "evidence.comparator_increment",
          "evidence.evaluation_mode",
          "evidence.interpretation_scope",
          "evidence.safety_reversibility",
          "evidence.user_control",
          "evidence.user_outcome",
          "route_capability"
        ],
        "decisive_values": {
          "deployment_stage": "REVERSIBLE_PERSONALIZATION",
          "evidence.comparator_increment": "SUPPORTED",
          "evidence.evaluation_mode": "IN_CONTEXT_USER_STUDY",
          "evidence.interpretation_scope": "MATCHED",
          "evidence.safety_reversibility": "REVERSIBLE",
          "evidence.user_control": "NOT_REPORTED",
          "evidence.user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "route_capability": "AVAILABLE"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "body_sensing_self_calibration",
        "source_id": "johnson2013calibration",
        "label_en": "User-shaped body-sensing calibration",
        "label_zh": "用户塑造的身体传感校准",
        "proposed_use_en": "Expose calibration so teachers and learners can shape goals and system behavior in the evaluated physical-skill setting.",
        "proposed_use_zh": "开放校准，使教师和学习者在已评估身体技能场景中塑造目标与系统行为。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "NOT_APPLICABLE",
          "evaluation_mode": "CASE_STUDY_IN_USE",
          "user_outcome": "DIRECT_USER_EXPERIENCE",
          "user_control": "SUPPORTED",
          "safety_reversibility": "USER_CONTROLLED",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "NOT_REQUIRED"
        },
        "evidence_basis_en": "Two in-use cases directly examined how exposed calibration supported ownership and goal setting rather than treating calibration as hidden setup.",
        "evidence_basis_zh": "两个真实使用案例直接考察了开放校准如何支持掌控感与目标设定，而非把校准当作隐藏设置。",
        "source_locator": "doi:10.1145/2493432.2493457#abstract",
        "naive_action": "HIDE_CALIBRATION_AS_TECHNICAL_SETUP"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "body_sensing_self_calibration",
        "source_id": "johnson2013calibration",
        "label_en": "User-shaped body-sensing calibration",
        "label_zh": "用户塑造的身体传感校准",
        "proposed_use_en": "Expose calibration so teachers and learners can shape goals and system behavior in the evaluated physical-skill setting.",
        "proposed_use_zh": "开放校准，使教师和学习者在已评估身体技能场景中塑造目标与系统行为。",
        "evidence_basis_en": "Two in-use cases directly examined how exposed calibration supported ownership and goal setting rather than treating calibration as hidden setup.",
        "evidence_basis_zh": "两个真实使用案例直接考察了开放校准如何支持掌控感与目标设定，而非把校准当作隐藏设置。",
        "source_locator": "doi:10.1145/2493432.2493457#abstract",
        "naive_action": "HIDE_CALIBRATION_AS_TECHNICAL_SETUP",
        "audited_action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        "audited_decision_en": "Proceed with personalization inside the evaluated boundary",
        "audited_decision_zh": "在已评估边界内实施个性化",
        "matched_rule_id": "R5_BOUNDED_PASS",
        "matched_rule_kind": "permission",
        "decisive_fields": [
          "deployment_stage",
          "evidence.comparator_increment",
          "evidence.evaluation_mode",
          "evidence.interpretation_scope",
          "evidence.safety_reversibility",
          "evidence.user_control",
          "evidence.user_outcome",
          "route_capability"
        ],
        "decisive_values": {
          "deployment_stage": "REVERSIBLE_PERSONALIZATION",
          "evidence.comparator_increment": "NOT_APPLICABLE",
          "evidence.evaluation_mode": "CASE_STUDY_IN_USE",
          "evidence.interpretation_scope": "MATCHED",
          "evidence.safety_reversibility": "USER_CONTROLLED",
          "evidence.user_control": "SUPPORTED",
          "evidence.user_outcome": "DIRECT_USER_EXPERIENCE",
          "route_capability": "AVAILABLE"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "model_based_adaptive_menu",
        "source_id": "todi2021modelbased",
        "label_en": "Conservative model-based menu adaptation",
        "label_zh": "保守型模型驱动 menu 自适应",
        "proposed_use_en": "Use the conservative policy in the evaluated adaptive-menu setting.",
        "proposed_use_zh": "在已评估自适应 menu 场景中使用保守策略。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "SUPPORTED",
          "evaluation_mode": "TECHNICAL_AND_EMPIRICAL_EVALUATION",
          "user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "user_control": "NOT_REPORTED",
          "safety_reversibility": "REVERSIBLE",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "NOT_REQUIRED"
        },
        "evidence_basis_en": "The conservative policy was designed to avoid changes when none were beneficial and outperformed non-adaptive and frequency-based policies in the reported evaluations.",
        "evidence_basis_zh": "保守策略被设计为在无益时不改变，并在报告评估中优于非自适应与频率策略。",
        "source_locator": "doi:10.1145/3411764.3445497#abstract",
        "naive_action": "REJECT_AUTOMATIC_ADAPTATION_CATEGORICALLY"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "model_based_adaptive_menu",
        "source_id": "todi2021modelbased",
        "label_en": "Conservative model-based menu adaptation",
        "label_zh": "保守型模型驱动 menu 自适应",
        "proposed_use_en": "Use the conservative policy in the evaluated adaptive-menu setting.",
        "proposed_use_zh": "在已评估自适应 menu 场景中使用保守策略。",
        "evidence_basis_en": "The conservative policy was designed to avoid changes when none were beneficial and outperformed non-adaptive and frequency-based policies in the reported evaluations.",
        "evidence_basis_zh": "保守策略被设计为在无益时不改变，并在报告评估中优于非自适应与频率策略。",
        "source_locator": "doi:10.1145/3411764.3445497#abstract",
        "naive_action": "REJECT_AUTOMATIC_ADAPTATION_CATEGORICALLY",
        "audited_action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        "audited_decision_en": "Proceed with personalization inside the evaluated boundary",
        "audited_decision_zh": "在已评估边界内实施个性化",
        "matched_rule_id": "R5_BOUNDED_PASS",
        "matched_rule_kind": "permission",
        "decisive_fields": [
          "deployment_stage",
          "evidence.comparator_increment",
          "evidence.evaluation_mode",
          "evidence.interpretation_scope",
          "evidence.safety_reversibility",
          "evidence.user_control",
          "evidence.user_outcome",
          "route_capability"
        ],
        "decisive_values": {
          "deployment_stage": "REVERSIBLE_PERSONALIZATION",
          "evidence.comparator_increment": "SUPPORTED",
          "evidence.evaluation_mode": "TECHNICAL_AND_EMPIRICAL_EVALUATION",
          "evidence.interpretation_scope": "MATCHED",
          "evidence.safety_reversibility": "REVERSIBLE",
          "evidence.user_control": "NOT_REPORTED",
          "evidence.user_outcome": "DIRECT_TASK_AND_EXPERIENCE",
          "route_capability": "AVAILABLE"
        },
        "decision_changed": true
      }
    },
    {
      "record": {
        "id": "interaction_data_visual_suggestions",
        "source_id": "alves2026interaction",
        "label_en": "Interaction-data visual suggestions",
        "label_zh": "交互数据可视建议",
        "proposed_use_en": "Deploy interaction-data-driven visual personalization suggestions with user acceptance and refinement.",
        "proposed_use_zh": "部署由交互数据驱动、可由用户接受和调整的可视个性化建议。",
        "deployment_stage": "REVERSIBLE_PERSONALIZATION",
        "route_capability": "AVAILABLE",
        "evidence": {
          "interpretation_scope": "MATCHED",
          "comparator_increment": "NOT_TESTED",
          "evaluation_mode": "VIGNETTE_ONLY",
          "user_outcome": "PREFERENCE_ONLY",
          "user_control": "SUPPORTED",
          "safety_reversibility": "USER_CONTROLLED",
          "persistence": "NOT_REQUIRED",
          "transfer": "NOT_REQUIRED",
          "reference_equity": "NOT_REQUIRED"
        },
        "evidence_basis_en": "Participants preferred supported, transparent suggestions in interviews, but the vignettes used synthetic data and did not test a functional system or task benefit.",
        "evidence_basis_zh": "访谈参与者偏好有支持且透明的建议，但 vignette 使用合成数据，未检验功能系统或任务收益。",
        "source_locator": "doi:10.1145/3772318.3791022#methods-and-conclusion",
        "naive_action": "DEPLOY_FROM_STATED_PREFERENCE"
      },
      "group": "external",
      "source_path": "configs/protocol/external_reuse_cases.json",
      "baseline": {
        "id": "interaction_data_visual_suggestions",
        "source_id": "alves2026interaction",
        "label_en": "Interaction-data visual suggestions",
        "label_zh": "交互数据可视建议",
        "proposed_use_en": "Deploy interaction-data-driven visual personalization suggestions with user acceptance and refinement.",
        "proposed_use_zh": "部署由交互数据驱动、可由用户接受和调整的可视个性化建议。",
        "evidence_basis_en": "Participants preferred supported, transparent suggestions in interviews, but the vignettes used synthetic data and did not test a functional system or task benefit.",
        "evidence_basis_zh": "访谈参与者偏好有支持且透明的建议，但 vignette 使用合成数据，未检验功能系统或任务收益。",
        "source_locator": "doi:10.1145/3772318.3791022#methods-and-conclusion",
        "naive_action": "DEPLOY_FROM_STATED_PREFERENCE",
        "audited_action": "RUN_PREREGISTERED_USER_STUDY",
        "audited_decision_en": "Run a preregistered in-context user study before deployment",
        "audited_decision_zh": "部署前开展预注册的情境内用户研究",
        "matched_rule_id": "R3_CONSEQUENCE_MATCH",
        "matched_rule_kind": "constraint",
        "decisive_fields": [
          "deployment_stage",
          "evidence.evaluation_mode",
          "evidence.user_outcome"
        ],
        "decisive_values": {
          "deployment_stage": "REVERSIBLE_PERSONALIZATION",
          "evidence.evaluation_mode": "VIGNETTE_ONLY",
          "evidence.user_outcome": "PREFERENCE_ONLY"
        },
        "decision_changed": true
      }
    }
  ]
};
