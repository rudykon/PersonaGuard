window.PERSONAGUARD_CASE_COPY = {
  "retention_transfer": {
    "expected_action": "WITHHOLD_RETENTION_AND_TRANSFER",
    "question": {
      "en": "Can we use this experiment’s profile again later?",
      "zh": "这次得到的用户画像，下次还能用吗？"
    },
    "use": {
      "en": "Save the profile from this experiment and use it in later sessions or other interfaces.",
      "zh": "保存本次实验得到的用户画像，在以后的实验或其他界面中继续使用。"
    },
    "evidence": {
      "en": "Stability across sessions and use across interfaces have not been tested. Coverage of the reference data cannot yet be adequately assessed.",
      "zh": "尚未检验画像在不同次实验中是否稳定，也未检验换到其他界面是否适用；参考数据对相关人群的覆盖情况也尚无法充分评估。"
    },
    "next": {
      "en": "Hold off on long-term storage and reuse in other settings.",
      "zh": "暂不长期保留，也不跨场景复用。"
    },
    "why": {
      "en": "Being able to compute a profile does not establish that it will still apply in a later session or another setting.",
      "zh": "能计算出画像，并不代表它在下次实验或其他场景仍然适用。"
    },
    "context": {
      "en": "This case concerns durable retention and transfer. It does not grant permission for a different use of the same profile.",
      "zh": "这个案例审查的是长期保留与跨场景复用，不会自动批准同一画像的其他用途。"
    },
    "summary_use": {
      "en": "Retain or transfer",
      "zh": "长期保留或迁移"
    },
    "summary_boundary": {
      "en": "Evidence for stability, transfer, and reference coverage is missing.",
      "zh": "稳定性、迁移与参考数据覆盖证据不足。"
    },
    "summary_action": {
      "en": "Withhold long-term retention and transfer.",
      "zh": "暂不长期保留或跨场景迁移。"
    },
    "review": {
      "en": "Evidence of persistence, transfer to the intended setting, and reference-data coverage for the intended population.",
      "zh": "补充长期稳定性、目标场景适用性，以及参考数据对目标人群覆盖的证据。"
    }
  },
  "optional_sensing": {
    "expected_action": "RETAIN_EVALUATED_COMPARATOR",
    "question": {
      "en": "Should we add sensors?",
      "zh": "要不要增加传感器？"
    },
    "use": {
      "en": "Add physiological sensors to predict the profile measured in the experiment.",
      "zh": "增加生理传感器，用它们来预测实验中测量的用户画像。"
    },
    "evidence": {
      "en": "In this measurement analysis, all evaluated sensor-based estimates had higher prediction error than the tested video-mean comparator.",
      "zh": "在这项测量分析中，所有已测试的传感方案，预测误差估计值都高于按视频取平均值的对照方案。"
    },
    "next": {
      "en": "Keep the tested video-mean comparator.",
      "zh": "保留已测试的视频均值方案。"
    },
    "why": {
      "en": "This analysis did not demonstrate a benefit from the additional sensing. That finding applies to this task and comparator; it does not establish that physiology has no useful information.",
      "zh": "这项分析没有证明额外采集生理信号能带来增益。结论只针对这项任务和对照，不能解释为生理信号没有有用信息。"
    },
    "context": {
      "en": "This case compares scalar timing-deviation prediction against video mean. The additional trajectory analysis above uses video-content prediction as a different comparator for a different target.",
      "zh": "本案例针对时间偏差量的预测，对照为视频均值。上方额外轨迹分析针对不同的目标，以视频内容预测作为另一种对照。"
    },
    "summary_use": {
      "en": "Add sensing",
      "zh": "增加传感"
    },
    "summary_boundary": {
      "en": "No demonstrated gain over the matched comparator for this target.",
      "zh": "在该任务中未证明优于匹配的对照方案。"
    },
    "summary_action": {
      "en": "Keep the evaluated comparator.",
      "zh": "保留已评估的对照方案。"
    },
    "review": {
      "en": "Demonstrated improvement over the timing-deviation comparator, using declared acquisition and burden criteria.",
      "zh": "在事先声明的采集与负担标准下，证明优于时间偏差量预测的对照方案。"
    }
  },
  "signed_calibration": {
    "expected_action": "RUN_PREREGISTERED_USER_STUDY",
    "question": {
      "en": "Can a better metric justify deployment?",
      "zh": "指标改善后，就能直接使用吗？"
    },
    "use": {
      "en": "Deploy a correction learned from separate calibration videos in a personalization system.",
      "zh": "把其他校准视频中得到的校正方法，部署到个性化系统中。"
    },
    "evidence": {
      "en": "The correction improved a measurement in the experiment, but user benefit has not been tested. How much improvement matters and what risks are acceptable remain to be established.",
      "zh": "这项校正改善了实验中的测量指标，但尚未检验用户能否受益。还没有确定：改善需要达到多大程度，以及哪些风险可以接受。"
    },
    "next": {
      "en": "Run a preregistered study in actual use before deployment.",
      "zh": "部署前，先开展预注册的真实使用研究。"
    },
    "why": {
      "en": "An improved measurement does not establish a better user outcome. The proposed use needs direct evidence of what happens to users in context.",
      "zh": "测量指标改善，并不能证明用户的体验或任务结果也会改善。还需要在真实使用情境中直接观察用户结果。"
    },
    "context": {
      "en": "The recorded proposal is a constant signed correction inferred from separate calibration videos. The audit distinguishes the trace endpoint from downstream user benefit.",
      "zh": "正式记录中的方案是从独立校准视频推断的常量有符号校正。审查区分了轨迹测量指标的改善和下游用户收益。"
    },
    "summary_use": {
      "en": "Deploy calibration",
      "zh": "部署校准"
    },
    "summary_boundary": {
      "en": "Reference error improved; direct user benefit was not tested.",
      "zh": "参考误差改善，尚未证明用户收益。"
    },
    "summary_action": {
      "en": "First run an in-context user study.",
      "zh": "先开展情境内用户研究。"
    },
    "review": {
      "en": "Direct evidence of benefit and degradation in actual use, with meaningful-gain and harm criteria specified in advance.",
      "zh": "事先确定有意义改善及伤害标准，再收集真实使用中收益与退化的直接证据。"
    }
  },
  "trace_interpretation": {
    "expected_action": "BOUND_TO_EVIDENCE_SCOPE",
    "question": {
      "en": "What can this result tell us?",
      "zh": "这个结果能说明什么？"
    },
    "use": {
      "en": "Interpret a profile calculated from the continuous self-report collected in one experiment.",
      "zh": "把一次实验中连续自评记录计算出的结果，解释为用户画像。"
    },
    "evidence": {
      "en": "The result changes with the interface, calculation method, and reference curve. It has not been checked in a repeat session.",
      "zh": "结果会随操作界面、计算方法和参考曲线变化，也没有通过另一次实验验证。"
    },
    "next": {
      "en": "Interpret it only within the tested settings and conditions.",
      "zh": "只在已测试的情境与条件下解释这个结果。"
    },
    "why": {
      "en": "This limits how the result can be interpreted and used. It neither approves personalization nor establishes a stable personal trait.",
      "zh": "这限定了可以如何解释和使用结果，不等于批准个性化，也不证明存在稳定的个人属性。"
    },
    "context": {
      "en": "The original measure is a post-trial temporal-deviation profile relative to a reference. R1 limits its scope; permission to personalize is a separate decision.",
      "zh": "原始测量是试次结束后相对参考曲线的时间偏差画像。R1 限定其范围，是否许可个性化需要另行判断。"
    },
    "summary_use": {
      "en": "Interpret the profile",
      "zh": "解释画像"
    },
    "summary_boundary": {
      "en": "Depends on the measurement and reference conditions.",
      "zh": "结果受测量与参考条件影响。"
    },
    "summary_action": {
      "en": "Limit interpretation and use.",
      "zh": "限定解释与使用范围。"
    },
    "review": {
      "en": "Evidence matching the proposed interpretation, interface, estimator, reference group, and intended reuse period.",
      "zh": "补充与解释目标、界面、估计方法、参考群体及复用期限匹配的验证。"
    }
  }
};
