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
      "source_locator": "results/evidence_traceability.json#eeg_fnirs_increment",
      "decision_owner_role": "sensor-analysis lead",
      "review_trigger": "Re-review after a materially different sensor representation, acquisition protocol, target, or prospective deployment setting is evaluated."
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
      "source_locator": "results/evidence_traceability.json#retention_and_transfer",
      "decision_owner_role": "data-governance lead",
      "review_trigger": "Re-review before any participant-level release, durable retention, cross-interface transfer, or consequential personalization decision."
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
      "source_locator": "results/evidence_traceability.json#signed_proximal_correction",
      "decision_owner_role": "personalization-analysis lead",
      "review_trigger": "Re-review before deployment and after preregistering a smallest worthwhile improvement, degradation limit, and consequential endpoint."
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
      "source_locator": "results/evidence_traceability.json#measurement_meaning",
      "decision_owner_role": "measurement-analysis lead",
      "review_trigger": "Re-review before any durable-profile claim or after a new session, interface, estimator, or cross-video transfer study."
    }
  ],
  "rules": [
    {
      "id": "R4_RETENTION_TRANSFER",
      "title_en": "Persistence, transfer, and reference coverage must match durable reuse",
      "title_zh": "持久性、迁移与参考覆盖必须匹配长期复用",
      "action": "WITHHOLD_RETENTION_AND_TRANSFER",
      "priority": 10,
      "kind": "constraint",
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
      }
    },
    {
      "id": "R2_COMPARATOR_INCREMENT",
      "title_en": "Added acquisition or adaptation must beat its route-matched comparator",
      "title_zh": "新增获取或自适应必须优于路径匹配比较方案",
      "action": "RETAIN_EVALUATED_COMPARATOR",
      "priority": 20,
      "kind": "constraint",
      "predicate": {
        "field": "evidence.comparator_increment",
        "eq": "NO_DEMONSTRATED_INCREMENT"
      }
    },
    {
      "id": "R3_CONSEQUENCE_MATCH",
      "title_en": "Evidence must match the consequence and be observed in use",
      "title_zh": "证据必须匹配后果并在真实使用中观察",
      "action": "RUN_PREREGISTERED_USER_STUDY",
      "priority": 30,
      "kind": "constraint",
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
      }
    },
    {
      "id": "R1_SCOPE_MATCH",
      "title_en": "Interpretation and use cannot exceed the evaluated evidence scope",
      "title_zh": "解释与用途不得超出已评估证据范围",
      "action": "BOUND_TO_EVIDENCE_SCOPE",
      "priority": 40,
      "kind": "constraint",
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
      }
    },
    {
      "id": "R5_BOUNDED_PASS",
      "title_en": "Matched direct evidence and reversible use can license bounded personalization",
      "title_zh": "匹配的直接证据与可逆使用可许可受限个性化",
      "action": "PROCEED_WITHIN_EVALUATED_BOUNDARY",
      "priority": 90,
      "kind": "permission",
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
      }
    }
  ],
  "resolver": {
    "policy": "Evaluate rules by ascending priority. The first matching constraint wins; a bounded-pass permission is returned only when no higher-priority constraint matches. No match returns a named evidence request, never silent approval.",
    "precedence_rationale": "Lifecycle obligations precede local performance; a failed route-matched comparator precedes consequence and scope; consequence mismatch precedes scope; positive permission is evaluated last. This harm-first order returns the highest-consequence unresolved obligation rather than treating priorities as empirical weights.",
    "fallback_action": "REQUIRE_ROUTE_SPECIFIC_EVIDENCE",
    "ablation_semantics": "Removing a rule removes its constraint or permission. An affected route becomes unresolved unless another declared rule independently matches; ablation does not assume automatic deployment."
  },
  "metrics": {
    "RevSixDenseExampleRanks": "91, 181, 270",
    "RevSixDenseExampleLengths": "122, 111, 122",
    "RevSixParticipants": "24",
    "RevSixTrials": "360",
    "RevSixSamples": "36,864",
    "RevSixCouplingGain": "0.806",
    "RevSixCouplingCI": "[0.305,1.406]",
    "RevSixVideoMeanMAE": "0.763",
    "RevSixContextMAE": "0.767",
    "RevSixStackMAE": "0.794",
    "RevSixStackGain": "-0.031",
    "RevSixStackGainCI": "[-0.081,0.014]",
    "RevSixAntiAliasMaxChange": "0.000272",
    "RevSixCVRepeats": "5",
    "RevSixOuterFolds": "5",
    "RevSixInnerFolds": "4",
    "RevSixDecisionThresholdPoints": "41",
    "RevSixDecisionThresholdMax": "0.200",
    "RevSixCVGainMin": "-0.031",
    "RevSixCVGainMax": "-0.001",
    "RevSixCVPositive": "0",
    "RevSixCVTotal": "5",
    "RevSixQCalibrationGains": "0.019/0.024/0.038/0.053",
    "RevSixQCalibrationCounts": "15/14/14/15",
    "RevSixSignedGainMin": "0.021",
    "RevSixSignedBudgetGains": "0.021/0.028/0.045/0.063",
    "RevSixSignedGainMax": "0.063",
    "RevSixSignedRelativeMin": "0.069",
    "RevSixSignedRelativeMax": "0.208",
    "RevSixProtocolCases": "4",
    "RevSixProtocolActions": "4",
    "RevSixProtocolChanged": "4",
    "RevSixProtocolOrderReplays": "100",
    "RevSixProtocolMutationRejects": "12",
    "RevSixProtocolLocalityPasses": "5",
    "RevSixProtocolPermissionProbes": "30",
    "RevSixProtocolPermissionPasses": "30",
    "RevSixProtocolPrecedenceProbes": "4",
    "RevSixProtocolPrecedencePasses": "4",
    "RevSixProtocolCombinedActions": "5",
    "RevSixProtocolRuleCount": "5",
    "RevSixProtocolMutationTotal": "12",
    "RevSixExternalSources": "6",
    "RevSixExternalCases": "7",
    "RevSixExternalActions": "3",
    "RevSixExternalPositive": "5",
    "RevSixExternalConstrained": "2",
    "RevSixExternalFallback": "0",
    "RevSixPythonVersion": "3.12.3",
    "RevSixNumpyVersion": "2.5.1",
    "RevSixStimulusFiles": "15",
    "RevSixStimulusOffsetMin": "1.700",
    "RevSixStimulusOffsetMax": "2.833",
    "RevSixAlgorithmOOFArtifacts": "12",
    "RevSixAlgorithmOOFArrays": "270",
    "RevSixMetadataPriorMAE": "31.333",
    "RevSixContentMAE": "30.493",
    "RevSixContentGain": "0.840",
    "RevSixContentRelativeGain": "2.68",
    "RevSixContentRepeatMean": "30.700",
    "RevSixContentRepeatSD": "0.208",
    "RevSixContentOffsetMin": "30.493",
    "RevSixContentOffsetMax": "30.736",
    "RevSixContentPositive": "5",
    "RevSixContentTotal": "5",
    "RevSixRouterMAE": "30.531",
    "RevSixRouterGain": "-0.037",
    "RevSixRouterPositive": "1",
    "RevSixRouterTotal": "5",
    "RevSixRouterRepeatMean": "30.726",
    "RevSixRouterRepeatSD": "0.179",
    "RevSixRouterOffsetMin": "30.485",
    "RevSixRouterOffsetMax": "30.626",
    "RevSixPhysioFullThreshold": "0.10",
    "RevSixPhysioFullMAE": "30.499",
    "RevSixPhysioFullGain": "-0.006",
    "RevSixPhysioTokenMAE": "30.493",
    "RevSixPhysioTokenGain": "0.000",
    "RevSixSparsePriorMAE": "28.417",
    "RevSixSparsePreviousMAE": "26.415",
    "RevSixSparseMAE": "26.289",
    "RevSixSparseGain": "2.127",
    "RevSixSparseRelativeGain": "7.49",
    "RevSixSparseGainPrevious": "0.126",
    "RevSixSparseGainCI": "[0.557,3.878]",
    "RevSixSparseFoldGainMin": "1.875",
    "RevSixSparseFoldGainMean": "1.980",
    "RevSixSparseFoldGainMax": "2.127",
    "RevSixSparsePositive": "5",
    "RevSixSparseTotal": "5",
    "RevSixContentCandidates": "97",
    "RevSixContentOuterSubjectFolds": "5",
    "RevSixContentOuterVideoFolds": "5",
    "RevSixContentInnerSubjectFolds": "2",
    "RevSixContentInnerVideoFolds": "3",
    "RevSixPhysioInnerSubjectFolds": "2",
    "RevSixPhysioInnerVideoFolds": "2",
    "RevSixSparseInnerFolds": "4",
    "RevSixSparseBootstrapRepeats": "5000",
    "PathMeanProfileRank": "0.497",
    "ReferenceMaxShift": "0.459",
    "SignedGainBudget1": "0.021",
    "SignedGainBudget2": "0.028",
    "SignedGainBudget4": "0.045",
    "SignedGainBudget8": "0.063"
  }
};
