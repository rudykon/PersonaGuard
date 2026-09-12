#!/usr/bin/env python3
"""Generate every publication-facing Revision-6 numerical consumer.

The canonical input is ``paper_support/revision6_source.json``.  This script owns the
manuscript table rows, LaTeX result macros, and marked README result blocks.
It deliberately has no fallback to historical artifact directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_support" / "revision6_source.json"
GENERATED = ROOT / "paper" / "generated"
START = "<!-- REVISION6_RESULTS:START -->"
END = "<!-- REVISION6_RESULTS:END -->"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def load_source() -> dict[str, object]:
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    if value.get("schema_version") != "revision6-publication-source-v1":
        raise RuntimeError("paper_support/revision6_source.json has the wrong schema")
    if value.get("revision") != 6:
        raise RuntimeError("Revision 6 must be the publication source")
    return value


def f(value: object, digits: int = 3) -> str:
    number = float(value)
    result = f"{abs(number):.{digits}f}"
    if result.startswith("0"):
        result = result[1:]
    if number < 0:
        result = "-" + result
    return result


def percent(value: object, digits: int = 3) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def tex_signed(value: object, digits: int = 3) -> str:
    result = f(value, digits)
    return rf"\({result}\)" if result.startswith("-") else result


def explicit_signed(value: object, digits: int = 3) -> str:
    result = f(value, digits)
    return result if result.startswith("-") else "+" + result


def ci(values: Iterable[object], digits: int = 3) -> str:
    low, high = list(values)
    return f"[{f(low, digits)},{f(high, digits)}]"


def rows_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {str(row[key]): row for row in rows}


def tex_text(value: object) -> str:
    """Escape plain registry text for LaTeX table cells."""

    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        text = text.replace(old, new)
    return text


def polish_chinese(value: object) -> str:
    """Translate residual English prose while retaining model and metric names."""

    text = str(value)
    for old, new in (
        ("participant-grouped", "按参与者分组的"),
        ("video mean", "视频均值"),
        ("split menu", "分区菜单"),
        ("model-driven menu adaptation", "模型驱动菜单适配"),
        ("vignette", "情境短文"),
        ("proxy", "代理指标"),
        ("session", "会话"),
    ):
        text = text.replace(old, new)
    return text


def signal_rows(source: dict[str, object], *, chinese: bool) -> str:
    rev4 = source["summaries"]["revision4"]
    sampling = rev4["sampling_rate_audit"]
    quality = rev4["signal_quality_audit"]
    eeg = quality["eeg"]
    fnirs = quality["fnirs"]
    fnirs_timeline = sampling["observed_video_samples_per_label_second"]["fnirs"]
    if chinese:
        rows = [
            (
                "EEG & 200 Hz & 每个标签秒恰好 1,000 个数组样本 & 恰好 5,000 个样本 & "
                "主流程在五倍降采样前采用 80-Hz 通带、100-Hz 阻带的严格 FFT 抗混叠；"
                "五点箱形平均用于实现敏感性分析。"
                f"平坦或极端通道的平均比例为 {f(eeg['flat_or_extreme_channel_ratio_mean'], 6)}；"
                f"{percent(eeg['seconds_with_any_flat_or_extreme_channel_fraction'], 2)}\\% 的秒至少含一个标记通道。"
                "分析尺度阈值为标准差低于 0.1 或峰峰值高于 500 $\\mu$V 等效单位。 \\\\"
            ),
            (
                "fNIRS & 47.62 Hz & "
                f"均值 {f(fnirs_timeline['mean'], 3)}；范围 {f(fnirs_timeline['minimum'], 3)}--"
                f"{f(fnirs_timeline['maximum'], 3)} 个样本/标签秒 & "
                "恰好 238 个样本 & 扣除五秒基线均值后线性插值到 4 Hz。"
                f"保留掩码保留 {fnirs['reserved_channels_range'][0]}--"
                f"{fnirs['reserved_channels_range'][1]} 个通道（均值 {fnirs['reserved_channels_mean']:.2f}）。"
                "运动尖峰定义为超过通道/类型一阶差分中位绝对值的 6 MAD；"
                f"平均比例为 {f(fnirs['motion_spike_ratio_mean'], 6)}，"
                f"{percent(fnirs['seconds_with_any_motion_spike_fraction'], 2)}\\% 的秒至少含一个标记尖峰。 \\\\"
            ),
        ]
    else:
        rows = [
            (
                "EEG & 200 Hz & Exactly 1,000 array samples per label second & Exactly 5,000 samples & "
                "Primary strict FFT anti-aliasing used an 80-Hz passband and 100-Hz stopband before decimation; "
                "five-sample boxcar reduction was the implementation sensitivity. "
                f"Mean flat-or-extreme channel ratio was {f(eeg['flat_or_extreme_channel_ratio_mean'], 6)}; "
                f"{percent(eeg['seconds_with_any_flat_or_extreme_channel_fraction'], 2)}\\% of seconds contained at least one flagged channel. "
                "Analysis-scale thresholds were s.d. below 0.1 or peak-to-peak above 500 $\\mu$V-equivalent units. \\\\"
            ),
            (
                "fNIRS & 47.62 Hz & "
                f"Mean {f(fnirs_timeline['mean'], 3)}; range {f(fnirs_timeline['minimum'], 3)}--"
                f"{f(fnirs_timeline['maximum'], 3)} samples per label second & "
                "Exactly 238 samples & Linear interpolation to 4 Hz after five-second baseline-mean subtraction. "
                f"Reservation masks retained {fnirs['reserved_channels_range'][0]}--"
                f"{fnirs['reserved_channels_range'][1]} channels (mean {fnirs['reserved_channels_mean']:.2f}). "
                "A motion spike exceeded the channel/type median absolute first difference by 6 MAD; "
                f"mean spike ratio was {f(fnirs['motion_spike_ratio_mean'], 6)} "
                f"and {percent(fnirs['seconds_with_any_motion_spike_fraction'], 2)}\\% of seconds contained "
                "at least one flagged reserved-channel spike. \\\\"
            ),
        ]
    return "\n".join([*rows, r"\bottomrule"]) + "\n"



def availability_rows(source: dict[str, object], *, chinese: bool) -> str:
    """Keep scalar and dense targets, timing, and validation boundaries distinct."""
    budgets = sorted(
        int(key)
        for key in source["summaries"]["revision5"][
            "signed_calibration_practical_value"
        ]["budgets"]
    )
    budget_text = "/".join(str(value) for value in budgets)
    if chinese:
        rows = [
            r"标量画像传感 & 试次级 $q$；MAE 单位为秒 & 完整试次离线生理信息；无目标摇杆轨迹输入 & 视频均值及无传感器情境；参与者分组嵌套验证 \\",
            rf"跨视频有符号校准 & 常量偏移后的参考近端轨迹误差 & {budget_text} 个独立校准视频报告 & 视频基线；校准与评价视频分开，参与者分组验证 \\",
            r"内容先验 & 稠密效价--唤醒度轨迹 & 完整目标视频可预处理；无用户反馈 & 元数据先验；参与者与已发布视频双重留出 \\",
            r"因果生理残差 & 内容先验之上的稠密轨迹残差 & 冻结内容及当前、过去生理样本 & 内容先验；参与者与已发布视频双重留出 \\",
            r"试后 SAM 恢复 & 已知视频的稠密轨迹重建 & 试后评分及训练参与者的同视频轨迹库 & 已知视频群体先验；仅参与者留出 \\",
        ]
    else:
        rows = [
            r"Trial-level timing prediction & Trial-level $q$; MAE in seconds & Complete-trial offline physiology; no target joystick trace input & Video mean and no-sensor context; participant-grouped nested validation \\",
            rf"Signed cross-video calibration & Reference-proximal trace error after constant correction & Reports from {budget_text} separate calibration videos & Video-only baseline; separate calibration/evaluation videos and participant-grouped validation \\",
            r"Content prior & Dense valence--arousal trajectory & Complete target video available for preprocessing; no user feedback & Metadata prior; participant and released-video double holdout \\",
            r"Causal physiology residual & Dense-trajectory residual above content & Frozen content and current/past physiological samples & Content prior; participant and released-video double holdout \\",
            r"Post-trial SAM recovery & Dense known-video reconstruction & Post-trial ratings and same-video training-participant trajectory library & Known-video population prior; participant holdout only \\",
        ]
    return "\n".join([*rows, r"\bottomrule"]) + "\n"



def sensing_rows(source: dict[str, object], *, chinese: bool) -> str:
    models = rows_by(source["tables"]["sensing_models"], "method")
    results = rows_by(source["tables"]["sensing_results"], "method")
    order = [
        "video_mean", "context_only", "cbramod_only", "handcrafted_eeg_only",
        "eeg_only", "fnirs_only", "eeg_fnirs_single_penalty", "context_eeg",
        "context_fnirs", "context_eeg_fnirs_single_penalty",
        "context_plus_blockwise_residual",
    ]
    input_translations = {
        "Video identity": "视频身份",
        "Fold-safe reference/video context": "折内安全的参考/视频情境",
        "EEG foundation features": "EEG 基础模型特征",
        "EEG spectral/Hjorth features": "EEG 频谱/Hjorth 特征",
        "CBraMod + handcrafted EEG": "CBraMod + 手工 EEG 特征",
        "Reservation-aware distributed-lag summaries": "保留掩码感知的分布滞后汇总",
        "Combined EEG + fNIRS": "组合 EEG + fNIRS",
        "Context + combined EEG": "情境 + 组合 EEG",
        "Context + fNIRS": "情境 + fNIRS",
        "Context + combined EEG + fNIRS": "情境 + 组合 EEG + fNIRS",
        "Context + inner-selected sensor residual (disabled allowed)": "情境 + 内层选择的传感残差（允许停用）",
    }
    model_translations = {
        "Video mean": "视频均值",
        "Context": "情境",
        "CBraMod": "CBraMod",
        "Handcrafted EEG": "手工 EEG",
        "Combined EEG": "组合 EEG",
        "fNIRS": "fNIRS",
        "EEG+fNIRS": "EEG+fNIRS",
        "Context+EEG": "情境+EEG",
        "Context+fNIRS": "情境+fNIRS",
        "Context+joint": "情境+联合传感",
        "Blockwise residual stack": "分块残差栈",
    }
    lines = []
    for method in order:
        model = models[method]
        result = results[method]
        inputs = (
            input_translations.get(model["inputs"], model["inputs"])
            if chinese
            else model["inputs"]
        )
        dimension = model["dimension"]
        if method == "video_mean":
            dimension = "---"
        elif dimension == "fold-selected / not fixed":
            dimension = "每折选择" if chinese else "fold-selected"
        elif str(dimension).isdigit():
            dimension = f"{int(dimension):,}"
        model_label = (
            model_translations.get(model["model"], model["model"])
            if chinese
            else model["model"]
        )
        lines.append(
            f"{model_label} & {inputs} & {dimension} & "
            f"{f(result['primary_antialias_reservation_mae_seconds'])} & "
            f"{tex_signed(result['gain_vs_video_mean_seconds'])} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"



def calibration_rows(source: dict[str, object], *, chinese: bool = False) -> str:
    budgets = source["summaries"]["revision5"][
        "signed_calibration_practical_value"
    ]["budgets"]
    lines = []
    for budget in sorted(budgets, key=int):
        row = budgets[budget]
        interval = row[
            "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95"
        ]
        burden_unit = "分钟" if chinese else "min"
        lines.append(
            f"{budget} & {f(row['no_correction_trace_mae_units'])} / "
            f"{f(row['video_only_trace_mae_units'])} / {f(row['calibrated_trace_mae_units'])} & "
            f"{f(row['calibrated_gain_vs_video_units'])} {ci(interval)} & "
            f"{f(row['calibrated_gain_vs_video_percent_of_video_mae'])}\\% & "
            f"{f(row['calibrated_oracle_gap_units'])} & "
            f"{row['participants_improved_vs_video']} / {row['participants_degraded_vs_video']} & "
            f"{tex_signed(row['worst_participant_gain_vs_video_units'])} & "
            f"{row['continuous_annotation_burden']['expected_minutes']:.1f} {burden_unit} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


def algorithm_route_rows(source: dict[str, object], *, chinese: bool) -> str:
    algorithms = source["summaries"]["algorithm_experiments"]
    zero = algorithms[
        "zero_interaction_held_out_participant_held_out_released_video"
    ]
    physiology = algorithms["causal_physiology_residual"]["full_modalities"]
    sparse = algorithms["one_post_trial_sam_sparse_recovery"]
    if chinese:
        lines = [
            (
                "内容条件化先验 & 仅刺激视频（零交互） & 元数据时间先验 & "
                f"{f(zero['selected_trial_macro_mae'])} & "
                f"{explicit_signed(zero['selected_gain_vs_metadata'])} & "
                "\\textbf{保留为有边界的无传感器基线} \\\\"
            ),
            (
                "完整模态因果残差 & 刺激视频 + 当前/过去 EEG/fNIRS & 冻结内容先验 & "
                f"{f(physiology['primary_trial_macro_mae'])} & "
                f"{explicit_signed(physiology['primary_gain'])} & "
                "\\textbf{不采用候选；停用残差后返回内容} \\\\"
            ),
            (
                "试后 SAM 稀疏恢复 & 每试次结束后一个效价--唤醒度 SAM 对 & "
                "已知视频群体先验 & "
                f"{f(sparse['gaussian_retrieval_ensemble_mae'])} & "
                f"{explicit_signed(sparse['gain_vs_canonical_prior'])} & "
                "\\textbf{仅限离线试后重建} \\\\"
            ),
        ]
    else:
        lines = [
            (
                "Content-conditioned prior & Stimulus video only (zero interaction) & "
                "Metadata temporal prior & "
                f"{f(zero['selected_trial_macro_mae'])} & "
                f"{explicit_signed(zero['selected_gain_vs_metadata'])} & "
                "\\textbf{Retain as a bounded no-sensor baseline} \\\\"
            ),
            (
                "Full-modality causal residual & Video plus current/past EEG--fNIRS & "
                "Frozen content prior & "
                f"{f(physiology['primary_trial_macro_mae'])} & "
                f"{explicit_signed(physiology['primary_gain'])} & "
                "\\textbf{Do not adopt candidate; disabled residual returns content} \\\\"
            ),
            (
                "Post-trial SAM sparse recovery & One valence--arousal SAM pair after each trial & "
                "Known-video population prior & "
                f"{f(sparse['gaussian_retrieval_ensemble_mae'])} & "
                f"{explicit_signed(sparse['gain_vs_canonical_prior'])} & "
                "\\textbf{Offline post-trial reconstruction only} \\\\"
            ),
        ]
    return "\n".join([*lines, r"\bottomrule"]) + "\n"



def algorithm_content_rows(
    source: dict[str, object], *, chinese: bool = False
) -> str:
    algorithms = source["summaries"]["algorithm_experiments"]
    zero = algorithms[
        "zero_interaction_held_out_participant_held_out_released_video"
    ]
    labels = {
        "clip": "CLIP",
        "siglip": "SigLIP",
        "dinov2": "DINOv2",
        "clip_siglip": (
            "等范数 CLIP+SigLIP" if chinese else "Equal-norm CLIP+SigLIP"
        ),
        "clip_siglip_weighted": (
            "CLIP+SigLIP（固定加权变体）"
            if chinese
            else "CLIP+SigLIP (fixed weighted variant)"
        ),
        "clip_siglip_delta1": (
            "CLIP+SigLIP（$\\delta=1$ 变体）"
            if chinese
            else "CLIP+SigLIP ($\\delta=1$ variant)"
        ),
        "clip_siglip_delta13": (
            "CLIP+SigLIP（$\\delta=1,3$ 变体）"
            if chinese
            else "CLIP+SigLIP ($\\delta=1,3$ variant)"
        ),
    }
    lines = []
    for row in zero["candidates"]:
        repeated = row["repeated_grouped_cv"]
        offset = row["offset_mae_range"]
        lines.append(
            f"{labels[row['name']]} & {f(row['trial_macro_mae'])} & "
            f"{explicit_signed(row['gain_vs_metadata'])} & "
            f"{f(repeated['mae_mean'])} $\\pm$ {f(repeated['mae_std'])} & "
            f"{repeated['positive_gain_runs']}/{repeated['runs']} & "
            f"{f(offset[0])}--{f(offset[1])} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"



def algorithm_physiology_rows(
    source: dict[str, object], *, chinese: bool = False
) -> str:
    physiology = source["summaries"]["algorithm_experiments"][
        "causal_physiology_residual"
    ]
    lines = []
    for key, label_en, label_zh in (
        ("full_modalities", "Full causal modalities", "完整因果模态"),
        (
            "temporal_token_modalities",
            "CBraMod temporal-token views",
            "CBraMod 时间令牌视图",
        ),
    ):
        label = label_zh if chinese else label_en
        route = physiology[key]
        primary = float(route["primary_minimum_inner_gain"])
        for row in route["decision_curve"]:
            threshold = float(row["minimum_inner_gain"])
            is_primary = abs(threshold - primary) < 1e-12
            primary_label = (
                ("是" if is_primary else "否")
                if chinese
                else ("Yes" if is_primary else "No")
            )
            lines.append(
                f"{label} & {f(threshold, 2)} & "
                f"{f(row['trial_macro_mae'])} & "
                f"{explicit_signed(row['gain_vs_content_prior'])} & "
                f"{primary_label} \\\\"
            )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"



def algorithm_sparse_rows(
    source: dict[str, object], *, chinese: bool = False
) -> str:
    sparse = source["summaries"]["algorithm_experiments"][
        "one_post_trial_sam_sparse_recovery"
    ]
    prior = float(sparse["canonical_prior_trial_macro_mae"])
    previous = float(sparse["previous_functional_ensemble_mae"])
    fold = sparse["fold_allocation_sensitivity"]
    if chinese:
        lines = [
            (
                f"规范已知视频群体先验 & {f(prior)} & .000 & --- & "
                "无试后个性化 \\\\"
            ),
            (
                f"既有函数型稀疏集成 & {f(previous)} & "
                f"{explicit_signed(prior - previous)} & --- & "
                "一个试后 SAM 对 \\\\"
            ),
            (
                "高斯检索 + 函数残差集成 & "
                f"{f(sparse['gaussian_retrieval_ensemble_mae'])} & "
                f"{explicit_signed(sparse['gain_vs_canonical_prior'])} & "
                f"{ci(sparse['participant_video_crossed_bootstrap_ci95'])} & "
                f"一个试后 SAM 对；划分增益 "
                f"{f(fold['gain_vs_canonical_prior_min'])}--"
                f"{f(fold['gain_vs_canonical_prior_max'])} \\\\"
            ),
        ]
    else:
        lines = [
            (
                f"Canonical known-video population prior & {f(prior)} & .000 & --- & "
                "No post-trial personalization \\\\"
            ),
            (
                f"Previous functional sparse ensemble & {f(previous)} & "
                f"{explicit_signed(prior - previous)} & --- & "
                "One post-trial SAM pair \\\\"
            ),
            (
                "Gaussian retrieval + functional residual ensemble & "
                f"{f(sparse['gaussian_retrieval_ensemble_mae'])} & "
                f"{explicit_signed(sparse['gain_vs_canonical_prior'])} & "
                f"{ci(sparse['participant_video_crossed_bootstrap_ci95'])} & "
                f"One post-trial SAM pair; split gains "
                f"{f(fold['gain_vs_canonical_prior_min'])}--"
                f"{f(fold['gain_vs_canonical_prior_max'])} \\\\"
            ),
        ]
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


NAIVE_LABELS = {
    "RETAIN_AS_DURABLE_PERSON_PROFILE": {
        "en": "Treat the computable score as a durable person profile",
        "zh": "把可计算分数视为持久个人画像",
    },
    "ADD_SENSORS_FOR_PERSONALIZATION": {
        "en": "Add sensors because a multimodal model can be fitted",
        "zh": "因可拟合多模态模型而增加传感器",
    },
    "DEPLOY_CALIBRATION": {
        "en": "Deploy from a positive proximal interval",
        "zh": "依据近端终点的正区间直接部署",
    },
    "RETAIN_AND_TRANSFER_PROFILE": {
        "en": "Store and transfer once the profile is computable",
        "zh": "画像可计算后即保存并迁移",
    },
}


def protocol_shortcut_rows(source: dict[str, object], *, chinese: bool) -> str:
    replay = source["summaries"]["protocol_replay"]
    by_id = {row["id"]: row for row in replay["case_results"]}
    order = [
        "trace_interpretation",
        "optional_sensing",
        "signed_calibration",
        "retention_transfer",
    ]
    suffix = "zh" if chinese else "en"
    lines = []
    for case_id in order:
        row = by_id[case_id]
        language = "zh" if chinese else "en"
        naive = NAIVE_LABELS[str(row["naive_action"])][language]
        label = row[f"label_{suffix}"]
        evidence = row[f"evidence_basis_{suffix}"]
        decision = row[f"audited_decision_{suffix}"]
        if chinese:
            label, evidence, decision = map(
                polish_chinese, (label, evidence, decision)
            )
        lines.append(
            f"{label} & "
            f"{naive} & "
            f"{evidence} & "
            f"\\textbf{{{decision}}} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


def protocol_replay_rows(source: dict[str, object], *, chinese: bool) -> str:
    """Render the frozen routes as uses, evidence, actions, and review obligations.

    The review column is an editorial summary of the existing rule contract;
    it is not a new case field, observation, or permission.
    """
    replay = source["summaries"]["protocol_replay"]
    suffix = "zh" if chinese else "en"
    by_id = {row["id"]: row for row in replay["case_results"]}
    review = {
        "trace_interpretation": (
            "Validation matching the proposed interpretation, interface, estimator/reference, and reuse horizon.",
            "与拟议解释、界面、估计器/参照及复用期限匹配的效度证据。",
        ),
        "optional_sensing": (
            "Demonstrated increment over the scalar-profile comparator under the declared acquisition and burden criteria.",
            "在声明的获取与负担判据下，证明优于标量画像比较方案的增量。",
        ),
        "signed_calibration": (
            "Direct in-context benefit and degradation evidence, with prespecified meaningful-gain and harm criteria.",
            "情境内直接收益与退化证据，并预设有意义增益及伤害判据。",
        ),
        "retention_transfer": (
            "Route-specific persistence, transfer, and intended-population reference-coverage evidence.",
            "路线特定的持续性、迁移及目标人群参照覆盖证据。",
        ),
    }
    lines = []
    for case_id, review_labels in review.items():
        row = by_id[case_id]
        label = row[f"label_{suffix}"]
        proposed = row[f"proposed_use_{suffix}"]
        evidence = row[f"evidence_basis_{suffix}"]
        decision = row[f"audited_decision_{suffix}"]
        if case_id == "trace_interpretation":
            decision = (
                "限定解释与用途"
                if chinese else "Bound interpretation and use"
            )
        if chinese:
            label, proposed, evidence, decision = map(
                polish_chinese, (label, proposed, evidence, decision)
            )
            proposed = proposed.replace("joystick", "摇杆")
        lines.append(
            f"{label}: {proposed} & {evidence} & "
            f"\\textbf{{{decision}}} & {review_labels[1 if chinese else 0]} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


def external_reuse_rows(source: dict[str, object], *, chinese: bool) -> str:
    replay = source["summaries"]["protocol_replay"]
    source_by_id = {
        row["id"]: row for row in replay["external_source_records"]
    }
    suffix = "zh" if chinese else "en"
    lines = []
    for row in replay["external_case_results"]:
        citation = source_by_id[row["source_id"]]["citation_key"]
        label = row[f"label_{suffix}"]
        evidence = row[f"evidence_basis_{suffix}"]
        decision = row[f"audited_decision_{suffix}"]
        if chinese:
            label, evidence, decision = map(
                polish_chinese, (label, evidence, decision)
            )
        lines.append(
            f"{label} \\cite{{{citation}}} & "
            f"{evidence} & "
            f"{str(row['matched_rule_id']).split('_', 1)[0]} & "
            f"\\textbf{{{decision}}} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


def protocol_rule_rows(source: dict[str, object], *, chinese: bool) -> str:
    replay = source["summaries"]["protocol_replay"]
    ablation = {row["rule_id"]: row for row in replay["rule_ablation"]}
    suffix = "zh" if chinese else "en"
    lines = []
    for row in replay["rule_inventory"]:
        citations = ",".join(row["derivation_sources"])
        title = row[f"title_{suffix}"]
        risk = row["risk_blocked_en"]
        kind = row["kind"]
        if chinese:
            risk = {
                "R1_SCOPE_MATCH": "防止把情境依赖分数扩张为稳定个人属性",
                "R2_COMPARATOR_INCREMENT": "防止仅因可拟合模型就接受额外复杂度或负担",
                "R3_CONSEQUENCE_MATCH": "防止把代理指标、偏好或情境短文反应改称用户收益",
                "R4_RETENTION_TRANSFER": "防止把可计算画像直接变成可持久或可迁移属性",
                "R5_BOUNDED_PASS": "防止审查只能阻断、不能许可有直接证据的个性化",
            }[row["id"]]
            kind = {"constraint": "约束", "permission": "许可"}[kind]
            title = polish_chinese(title)
        lines.append(
            f"{str(row['id']).split('_', 1)[0]} & {kind} & {title} & "
            f"{risk} & \\cite{{{citations}}} & "
            f"{ablation[row['id']]['cases_changed']} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


def protocol_contract_rows(source: dict[str, object], *, chinese: bool) -> str:
    """Render the five-row main-text contract in actual execution order."""

    inventory = source["summaries"]["protocol_replay"]["rule_inventory"]
    expected = [
        ("R4_RETENTION_TRANSFER", 10, "WITHHOLD_RETENTION_AND_TRANSFER"),
        ("R2_COMPARATOR_INCREMENT", 20, "RETAIN_EVALUATED_COMPARATOR"),
        ("R3_CONSEQUENCE_MATCH", 30, "RUN_PREREGISTERED_USER_STUDY"),
        ("R1_SCOPE_MATCH", 40, "BOUND_TO_EVIDENCE_SCOPE"),
        ("R5_BOUNDED_PASS", 90, "PROCEED_WITHIN_EVALUATED_BOUNDARY"),
    ]
    observed = [
        (str(row["id"]), int(row["priority"]), str(row["action"]))
        for row in inventory
    ]
    if observed != expected:
        raise RuntimeError("The frozen rule-contract order or action has changed")

    if chinese:
        cells = {
            "R4_RETENTION_TRANSFER": (
                "阶段为保留/迁移，且持久性或迁移为部分/未检验，或参照公平为部分/不可评估",
                "暂不进行持久保留与跨情境迁移",
                "把可计算画像固化为持久、可携带的个人属性",
                "补充路径特定的持久性、迁移及目标人群参照覆盖/公平证据；任何“无需评价”状态均须明确论证",
            ),
            "R2_COMPARATOR_INCREMENT": (
                "相对路径匹配比较方案未证明增量",
                "保留已评价的路径匹配比较方案",
                "仅因模型可拟合就接受新增复杂度、负担、采集或自动改变",
                "取得足以修订“未证明增量”状态的路径匹配比较证据，并采用该路径声明的收益与负担判据",
            ),
            "R3_CONSEQUENCE_MATCH": (
                "阶段为可逆或后果性个性化，且评价仅为代理/情境短文/未检验，或结果仅为近端指标/偏好/未检验",
                "部署前开展预注册的情境内用户研究",
                "把代理修正、偏好或情境短文反应改称已证明的用户收益",
                "取得预注册的情境内或真实使用证据，直接测量任务/体验结果以及与后果匹配的收益和伤害",
            ),
            "R1_SCOPE_MATCH": (
                "路径能力可用，但解释范围仅部分匹配或未检验",
                "把解释与用途限定在已评价证据范围内",
                "把情境依赖分数扩张为稳定个人属性",
                "取得与拟议解释、人群、任务、界面、估计器/参照和用途匹配的效度证据，使范围达到匹配",
            ),
            "R5_BOUNDED_PASS": (
                "可逆个性化；能力可用、范围匹配、比较增量获支持/部分支持/不适用；有使用中直接结果，且控制获支持或改变可逆/由用户控制",
                "仅在已评价边界内实施个性化",
                "防止审计退化为只能约束而不能许可有证据路径的单向清单",
                "六个边界维度必须持续成立；人群、任务、界面、后果或信息通道改变即创建新路径并重新审计",
            ),
        }
    else:
        cells = {
            "R4_RETENTION_TRANSFER": (
                "Retention/transfer; persistence or transfer is partial/untested, or reference equity is partial/not assessable",
                "Withhold durable retention and cross-context transfer",
                "A computable profile becomes a durable or portable person property",
                "Route-specific persistence, transfer, and intended-population reference-coverage/equity evidence; justify any not-required field",
            ),
            "R2_COMPARATOR_INCREMENT": (
                "Route-matched comparator shows no demonstrated increment",
                "Retain the evaluated route-matched comparator",
                "Added complexity, burden, acquisition, or automatic change is accepted because a model can be fitted",
                "Route-matched evidence sufficient to revise the no-increment finding, using the declared gain and burden criterion",
            ),
            "R3_CONSEQUENCE_MATCH": (
                "Reversible or consequential personalization; proxy/vignette/untested evaluation, or proximal/preference/untested outcome",
                "Run a preregistered in-context user study before deployment",
                "A proxy correction, preference, or vignette response is relabeled as demonstrated user benefit",
                "Preregistered in-context or in-use evidence with direct task/experience and consequence-matched benefit and harm endpoints",
            ),
            "R1_SCOPE_MATCH": (
                "Capability available, but interpretation scope is partial or untested",
                "Bound interpretation and use to the evaluated evidence scope",
                "A context-dependent score is promoted to a stable person property",
                "Validation matching interpretation, population, task, interface, estimator/reference, and use; recode scope as matched",
            ),
            "R5_BOUNDED_PASS": (
                "Reversible personalization; available capability, matched scope, supported/partial/N.A. increment, observed-in-use direct outcome, and supported control or reversible/user-controlled change",
                "Proceed only inside the evaluated boundary",
                "The audit can constrain routes but can never authorize a supported one",
                "All six boundary dimensions must remain satisfied; changed population, task, interface, consequence, or information channel creates a new route",
            ),
        }

    lines = []
    for row in inventory:
        trigger, action, blocked, rereview = cells[str(row["id"])]
        short_id = str(row["id"]).split("_", 1)[0]
        lines.append(
            f"\\textbf{{{short_id}/{int(row['priority'])}}} & "
            f"{trigger} & \\textbf{{{action}}} & {blocked} & {rereview} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"



def adjacent_framework_rows(
    source: dict[str, object], *, chinese: bool = False
) -> str:
    comparison = source["summaries"]["adjacent_frameworks"]
    translations = {
        "Model Cards": (
            "模型卡（Model Cards）",
            "已发布模型",
            "预期用途与基准报告",
            "记录适用性边界；路径解析不是其主要输出",
            "模型卡实例",
        ),
        "Datasheets for Datasets": (
            "数据集说明书（Datasheets for Datasets）",
            "已发布数据集",
            "动机、构成、收集与使用说明",
            "为下游选择提供信息；候选路径动作不是其主要输出",
            "模板论证与示例",
        ),
        "Co-designed Fairness Checklists": (
            "共创公平性检查表",
            "组织开发实践",
            "检查表、组织诉求与误用风险",
            "支持流程反思；不声称提供带类型的路径解析器",
            "与 48 名从业者迭代共创",
        ),
        "End-to-End Internal Algorithmic Auditing": (
            "端到端内部算法审计",
            "组织级 AI 生命周期",
            "生命周期审计框架与文档链",
            "连接审计阶段与问责；并非专为并行 HCI 画像路径设计",
            "流程综合与组织案例依据",
        ),
        "Human--AI Interaction Guidelines": (
            "人机 AI 交互指南",
            "面向人的 AI 交互",
            "18 条设计指南",
            "指导设计选择；来源关联的路径动作不是其主要输出",
            "多轮研究，包括 49 名从业者和 20 个产品",
        ),
        "This route-audit protocol": (
            "本文路径审计协议",
            "候选画像的解释、获取、行动、保留或迁移路径",
            "带类型证据记录、路径特定动作与复审触发条件",
            "声明式约束/许可解析器，显式记录比较基线、后果、负担、范围与生命周期",
            "工作案例重放、非法记录与局部性测试、六来源外部迁移基准和五项规则消融",
        ),
    }
    lines = []
    for row in comparison["rows"]:
        framework = row["framework"]
        if chinese:
            display, unit, output, relation, basis = translations[framework]
        else:
            display = framework
            unit = row["unit"]
            output = row["primary_output"]
            relation = row["evidence_to_action"]
            basis = row["evaluation_basis"]
        if row["citation_key"] != "self":
            display += f" \\cite{{{row['citation_key']}}}"
        lines.append(
            f"{display} & {unit} & {output} & {relation} & {basis} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"



def statistical_registry_rows(
    source: dict[str, object], *, chinese: bool = False
) -> str:
    family_labels = {
        "formula_target_audit": "Formula and target audit",
        "same_trial_cross_axis": "Same-trial cross-axis",
        "cross_video_transfer": "Cross-video transfer",
        "category_heterogeneity": "Category heterogeneity",
        "estimator_reference_sensitivity": "Estimator/reference sensitivity",
        "optional_sensing_increment": "Optional sensing increment",
        "signed_calibration": "Signed calibration",
        "decision_threshold_curves": "Decision-threshold curves",
        "content_conditioned_trajectory": "Content-conditioned trajectory",
        "causal_physiology_residual": "Causal physiology residual",
        "post_trial_sam_sparse_recovery": "Post-trial SAM sparse recovery",
    }
    chinese_rows = {
        "formula_target_audit": {
            "family": "公式与目标审计",
            "endpoint": "已保存与独立实现的 b、q 目标；q 不小于 |b| 的结构不变量",
            "unit": "360 个参与者--视频试次（结构身份检查）",
            "estimator": "精确重算以及相等性/不变量检查",
            "uncertainty": "无；这是确定性工件检查",
            "folds": "无",
            "multiplicity": "不适用",
            "status": "结构检查",
        },
        "same_trial_cross_axis": {
            "family": "同试次跨轴分析",
            "endpoint": "另一轴对齐增益，以及自身相对供体时间扭曲的优势",
            "unit": "参与者；视频作为发布支持点交叉处理",
            "estimator": "均值增益、参与者×视频交叉自助法、参与者符号翻转诊断和供体错配",
            "uncertainty": "百分位交叉自助区间；10,000 次符号翻转/错配抽样",
            "folds": "5,000 次自助抽样；保留外层参与者折",
            "multiplicity": "分轴及组合对比均为探索性；不作族错误率主张",
            "status": "探索性界面依赖诊断",
        },
        "cross_video_transfer": {
            "family": "跨视频迁移",
            "endpoint": "留出视频上的有符号残差或绝对幅值预测增益",
            "unit": "参与者；视频在参与者安全折内留出",
            "estimator": "针对 1/2/4/8 个视频预算的嵌套收缩校准",
            "uncertainty": "5,000 次参与者百分位自助抽样",
            "folds": "每个预算 100 个校准子集",
            "multiplicity": "四种预算作为探索性响应曲线报告；不作确证性阈值主张",
            "status": "探索性迁移分析",
        },
        "category_heterogeneity": {
            "family": "类别异质性",
            "endpoint": "五个类别平均增益之间的平方偏差和",
            "unit": "参与者区组；发布视频类别",
            "estimator": "参与者内类别标签置换",
            "uncertainty": "由 10,000 次置换得到的蒙特卡洛 p 值",
            "folds": "参与者分块标签置换",
            "multiplicity": "各预算 p 值在四种预算间作 Holm 校正；另报合并全部预算的诊断",
            "status": "探索性异质性检验",
        },
        "estimator_reference_sensitivity": {
            "family": "估计器/参考敏感性",
            "endpoint": "画像排序、q 位移、路径几何和共享参照描述性方差比",
            "unit": "声明经验支持下的参与者和发布视频",
            "estimator": "五种估计目标、四种路径目标、参数及参考面板敏感性",
            "uncertainty": "排除原始身份的全流程自助法（1,000 次）；几何自助法（1,000 次）",
            "folds": "参与者簇重采样、类别内视频抽样并重建参考",
            "multiplicity": "敏感性清单；不设可靠性阈值或确证性检验族",
            "status": "描述性敏感性",
        },
        "optional_sensing_increment": {
            "family": "可选传感增量",
            "endpoint": "相对视频均值与仅情境比较基线的参与者宏平均 MAE 增益",
            "unit": "外层留出参与者",
            "estimator": "嵌套分组交叉验证，折内安全地构造目标、特征、缩放、选择及停用残差选项",
            "uncertainty": "5,000 次参与者百分位自助抽样，加完整划分敏感性",
            "folds": "5 个外层×4 个内层折；5 次完整参与者划分",
            "multiplicity": "模型清单为描述性；分块残差栈是声明的主要传感路径；无等效界值",
            "status": "主要路径比较，未预注册",
        },
        "signed_calibration": {
            "family": "有符号校准",
            "endpoint": "相对仅视频校正的参考轨迹 MAE 增益",
            "unit": "参与者",
            "estimator": "外层训练收缩，使用 1/2/4/8 个校准视频预算",
            "uncertainty": "5,000 次参与者百分位自助抽样",
            "folds": "每个预算 100 个校准子集；参与者安全折",
            "multiplicity": "四种预算形成探索性负担--响应曲线；未检验最小有意义增益或效用",
            "status": "近端证据，不是用户效用证据",
        },
        "decision_threshold_curves": {
            "family": "决策阈值曲线",
            "endpoint": "超过假设收益或退化阈值的参与者人数",
            "unit": "24 个参与者级估计增益",
            "estimator": "固定 41 点网格上的精确整数计数",
            "uncertainty": "无区间；有意移除条件式 Wilson 带",
            "folds": "使用已登记传感/校准流程的参与者估计",
            "multiplicity": "仅作事后描述性敏感性；阈值不定义决策规则",
            "status": "描述性决策曲线",
        },
        "content_conditioned_trajectory": {
            "family": "内容条件化轨迹",
            "endpoint": "1--255 量表上的试次宏平均效价/唤醒度 MAE，以及相对元数据时间先验的增益",
            "unit": "参与者；视频作为固定发布支持点并在外层单元中联合留出",
            "estimator": "冻结视觉表征、同类别轨迹检索、单调内容对齐和内层选择的精确回退",
            "uncertainty": "五次完整分组交叉验证参与者划分，加固定 -3 至 +3 秒对齐敏感性范围",
            "folds": "5 个参与者×5 个视频外层单元；2 个参与者×3 个视频内层折；5 次参与者折分配",
            "multiplicity": "七个已报告骨干/融合候选构成开发清单；不作确证性胜者主张",
            "status": "迭代开发集嵌套 OOF 估计",
        },
        "causal_physiology_residual": {
            "family": "因果生理残差",
            "endpoint": "相对冻结 CLIP+SigLIP 内容预测的试次宏平均 MAE 增益",
            "unit": "参与者；发布视频在固定 15 视频支持内联合留出",
            "estimator": "仅使用当前/过去 EEG--fNIRS 的 Ridge 残差，带内层增益门控和精确零残差回退",
            "uncertainty": "冻结五点内层增益决策曲线；无等效区间或用户效用主张",
            "folds": "与内容先验相同的 5 个参与者×5 个视频外层单元；2 个参与者×2 个视频内层折",
            "multiplicity": "完整特征与时间令牌路径作为门控开发比较报告",
            "status": "主要门控失败；精确回退到内容",
        },
        "post_trial_sam_sparse_recovery": {
            "family": "试后 SAM 稀疏恢复",
            "endpoint": "获得一个试后 SAM 对后，相对规范已知视频群体先验的试次宏平均 MAE 与增益",
            "unit": "参与者；不确定性传播中视频作为固定发布支持交叉处理",
            "estimator": "高斯加权的同视频轨迹检索，加有界中心/形状校正与函数残差集成",
            "uncertainty": "5,000 次参与者--视频交叉百分位自助抽样及完整参与者折分配敏感性",
            "folds": "5 个外层参与者折×4 个内层折；5 次完整分配",
            "multiplicity": "候选清单与集成在本数据集上开发；需要独立确认",
            "status": "试后稀疏反馈路径；不是零交互",
        },
    }
    lines = []
    for row in source["statistical_analysis_registry"]["rows"]:
        if chinese:
            translated = chinese_rows[row["id"]]
            family = translated["family"]
            endpoint = translated["endpoint"]
            population_unit = translated["unit"]
            estimator = translated["estimator"]
            uncertainty = translated["uncertainty"]
            folds = translated["folds"]
            multiplicity = translated["multiplicity"]
            status = translated["status"]
            labels = {
                "endpoint": "终点",
                "unit": "单位",
                "estimator": "估计器/检验",
                "uncertainty": "不确定性",
                "folds": "折分/重采样",
                "multiplicity": "多重性",
                "seed": "种子",
                "status": "状态",
            }
        else:
            family = family_labels[row["id"]]
            endpoint = row["endpoint"]
            population_unit = row["population_unit"]
            estimator = row["estimator_or_test"]
            uncertainty = row["uncertainty"]
            folds = row["resampling_or_folds"]
            multiplicity = row["multiplicity"]
            status = row["status"]
            labels = {
                "endpoint": "Endpoint",
                "unit": "Unit",
                "estimator": "Estimator/test",
                "uncertainty": "Uncertainty",
                "folds": "Folds/resampling",
                "multiplicity": "Multiplicity",
                "seed": "Seed",
                "status": "Status",
            }
        lines.append(
            f"{family} & "
            f"\\textbf{{{labels['endpoint']}:}} {tex_text(endpoint)} \\newline "
            f"\\textbf{{{labels['unit']}:}} {tex_text(population_unit)} & "
            f"\\textbf{{{labels['estimator']}:}} {tex_text(estimator)} \\newline "
            f"\\textbf{{{labels['uncertainty']}:}} {tex_text(uncertainty)} \\newline "
            f"\\textbf{{{labels['folds']}:}} {tex_text(folds)} & "
            f"\\textbf{{{labels['multiplicity']}:}} {tex_text(multiplicity)} \\newline "
            f"\\textbf{{{labels['seed']}:}} {tex_text(row['seed'])} \\newline "
            f"\\textbf{{{labels['status']}:}} {tex_text(status)} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule"]) + "\n"


def macros(source: dict[str, object], *, chinese: bool = False) -> str:
    rev4 = source["summaries"]["revision4"]
    rev5 = source["summaries"]["revision5"]
    rev6 = source["summaries"]["revision6"]
    sensing = rev6["primary_antialias_reservation_sensing"]
    repeated = sensing["repeated_grouped_cv_sensitivity"]
    repeated_gain = repeated["primary_route_gain_vs_video_mean_seconds"]
    threshold_grid = rev6["decision_threshold_curves"]["threshold_grid"]
    results = rows_by(source["tables"]["sensing_results"], "method")
    q_calibration = rev4["estimand_calibration"]["consensus"]
    signed = rev5["signed_calibration_practical_value"]["budgets"]
    anti_alias = sensing[
        "maximum_absolute_mae_change_eeg_affected_methods_seconds"
    ]
    stack = results["context_plus_blockwise_residual"]
    video = results["video_mean"]
    context = results["context_only"]
    coupling = source["summaries"]["revision2"]["own_other_warp"]["combined"]
    replay = source["summaries"]["protocol_replay"]["validation"]
    external = source["summaries"]["protocol_replay"]["external_transfer"]
    software = source["statistical_analysis_registry"]["software"]
    algorithms = source["summaries"]["algorithm_experiments"]
    zero = algorithms[
        "zero_interaction_held_out_participant_held_out_released_video"
    ]
    router = zero["router"]
    physiology = algorithms["causal_physiology_residual"]
    physiology_full = physiology["full_modalities"]
    physiology_tokens = physiology["temporal_token_modalities"]
    sparse = algorithms["one_post_trial_sam_sparse_recovery"]
    sparse_fold = sparse["fold_allocation_sensitivity"]
    stimulus = algorithms["stimulus_integrity"]
    algorithm_oof = algorithms["oof_artifact_consistency"]
    content_config = source["summaries"]["av_content_run_manifest"][
        "configuration"
    ]
    physiology_config = source["summaries"]["av_physio_run_manifest"][
        "configuration"
    ]
    sparse_config = source["summaries"]["sparse_anchor_run_manifest"][
        "configuration"
    ]
    values = {
        "RevSixDenseExampleRanks": ("、" if chinese else ", ").join(
            str(e["rank"]) for e in source["summaries"]["dense_trajectory_examples"]["examples"]
        ),
        "RevSixDenseExampleLengths": ("、" if chinese else ", ").join(
            str(len(e["time_seconds"]))
            for e in source["summaries"]["dense_trajectory_examples"]["examples"]
        ),
        "RevSixParticipants": "24",
        "RevSixTrials": "360",
        "RevSixSamples": "36,864",
        "RevSixCouplingGain": f(coupling["mean"]),
        "RevSixCouplingCI": ci(coupling["participant_video_crossed_ci95"]),
        "RevSixVideoMeanMAE": f(video["primary_antialias_reservation_mae_seconds"]),
        "RevSixContextMAE": f(context["primary_antialias_reservation_mae_seconds"]),
        "RevSixStackMAE": f(stack["primary_antialias_reservation_mae_seconds"]),
        "RevSixStackGain": f(stack["gain_vs_video_mean_seconds"]),
        "RevSixStackGainCI": ci([stack["gain_vs_video_mean_ci95_low"], stack["gain_vs_video_mean_ci95_high"]]),
        "RevSixAntiAliasMaxChange": f(anti_alias, 6),
        "RevSixCVRepeats": str(repeated["repeats"]),
        "RevSixOuterFolds": str(repeated["outer_folds_per_repeat"]),
        "RevSixInnerFolds": str(repeated["inner_folds"]),
        "RevSixDecisionThresholdPoints": str(len(threshold_grid)),
        "RevSixDecisionThresholdMax": f(max(threshold_grid), 3),
        "RevSixCVGainMin": f(repeated_gain["minimum_across_partitions"]),
        "RevSixCVGainMax": f(repeated_gain["maximum_across_partitions"]),
        "RevSixCVPositive": str(repeated_gain["partitions_positive"]),
        "RevSixCVTotal": str(repeated_gain["partitions_total"]),
        "RevSixQCalibrationGains": "/".join(f(q_calibration[str(b)]["mean_gain_seconds"]) for b in (1, 2, 4, 8)),
        "RevSixQCalibrationCounts": "/".join(str(q_calibration[str(b)]["participants_improved"]) for b in (1, 2, 4, 8)),
        "RevSixSignedGainMin": f(signed["1"]["calibrated_gain_vs_video_units"]),
        "RevSixSignedBudgetGains": "/".join(
            f(signed[str(b)]["calibrated_gain_vs_video_units"])
            for b in (1, 2, 4, 8)
        ),
        "RevSixSignedGainMax": f(signed["8"]["calibrated_gain_vs_video_units"]),
        "RevSixSignedRelativeMin": f(signed["1"]["calibrated_gain_vs_video_percent_of_video_mae"]),
        "RevSixSignedRelativeMax": f(signed["8"]["calibrated_gain_vs_video_percent_of_video_mae"]),
        "RevSixProtocolCases": str(replay["case_count"]),
        "RevSixProtocolActions": str(replay["distinct_audited_actions"]),
        "RevSixProtocolChanged": str(replay["naive_decisions_changed"]),
        "RevSixProtocolOrderReplays": str(replay["order_permutations"]),
        "RevSixProtocolMutationRejects": str(replay["invalid_mutations_rejected"]),
        "RevSixProtocolLocalityPasses": str(replay["route_locality_probes_passed"]),
        "RevSixProtocolPermissionProbes": str(replay["permission_boundary_probes"]),
        "RevSixProtocolPermissionPasses": str(replay["permission_boundary_probes_passed"]),
        "RevSixProtocolPrecedenceProbes": str(replay["precedence_probes"]),
        "RevSixProtocolPrecedencePasses": str(replay["precedence_probes_passed"]),
        "RevSixProtocolCombinedActions": str(replay["combined_distinct_actions"]),
        "RevSixProtocolRuleCount": str(replay["rule_ablations"]),
        "RevSixProtocolMutationTotal": str(replay["invalid_mutations"]),
        "RevSixExternalSources": str(external["source_count"]),
        "RevSixExternalCases": str(external["case_count"]),
        "RevSixExternalActions": str(external["distinct_audited_actions"]),
        "RevSixExternalPositive": str(external["positive_bounded_actions"]),
        "RevSixExternalConstrained": str(external["constrained_or_study_actions"]),
        "RevSixExternalFallback": str(external["fallback_actions"]),
        "RevSixPythonVersion": str(software["python"]),
        "RevSixNumpyVersion": str(software["numpy"]),
        "RevSixStimulusFiles": str(stimulus["files_present"]),
        "RevSixStimulusOffsetMin": f(
            stimulus["media_minus_annotation_seconds"]["minimum"]
        ),
        "RevSixStimulusOffsetMax": f(
            stimulus["media_minus_annotation_seconds"]["maximum"]
        ),
        "RevSixAlgorithmOOFArtifacts": str(algorithm_oof["artifacts_checked"]),
        "RevSixAlgorithmOOFArrays": str(algorithm_oof["numeric_arrays_checked"]),
        "RevSixMetadataPriorMAE": f(zero["metadata_prior_trial_macro_mae"]),
        "RevSixContentMAE": f(zero["selected_trial_macro_mae"]),
        "RevSixContentGain": f(zero["selected_gain_vs_metadata"]),
        "RevSixContentRelativeGain": f(
            100.0
            * zero["selected_gain_vs_metadata"]
            / zero["metadata_prior_trial_macro_mae"],
            2,
        ),
        "RevSixContentRepeatMean": f(zero["selected_repeat_mae_mean"]),
        "RevSixContentRepeatSD": f(zero["selected_repeat_mae_std"]),
        "RevSixContentOffsetMin": f(zero["selected_offset_mae_range"][0]),
        "RevSixContentOffsetMax": f(zero["selected_offset_mae_range"][1]),
        "RevSixContentPositive": str(
            next(
                row["repeated_grouped_cv"]["positive_gain_runs"]
                for row in zero["candidates"]
                if row["name"] == zero["selected_model"]
            )
        ),
        "RevSixContentTotal": str(
            next(
                row["repeated_grouped_cv"]["runs"]
                for row in zero["candidates"]
                if row["name"] == zero["selected_model"]
            )
        ),
        "RevSixRouterMAE": f(router["trial_macro_mae"]),
        "RevSixRouterGain": f(router["gain_vs_frozen_baseline"]),
        "RevSixRouterPositive": str(router["positive_repeat_runs"]),
        "RevSixRouterTotal": str(router["total_repeat_runs"]),
        "RevSixRouterRepeatMean": f(router["repeated_mae_mean"]),
        "RevSixRouterRepeatSD": f(router["repeated_mae_std"]),
        "RevSixRouterOffsetMin": f(router["offset_mae_range"][0]),
        "RevSixRouterOffsetMax": f(router["offset_mae_range"][1]),
        "RevSixPhysioFullThreshold": f(
            physiology_full["primary_minimum_inner_gain"], 2
        ),
        "RevSixPhysioFullMAE": f(
            physiology_full["primary_trial_macro_mae"]
        ),
        "RevSixPhysioFullGain": f(physiology_full["primary_gain"]),
        "RevSixPhysioTokenMAE": f(
            physiology_tokens["primary_trial_macro_mae"]
        ),
        "RevSixPhysioTokenGain": f(physiology_tokens["primary_gain"]),
        "RevSixSparsePriorMAE": f(
            sparse["canonical_prior_trial_macro_mae"]
        ),
        "RevSixSparsePreviousMAE": f(
            sparse["previous_functional_ensemble_mae"]
        ),
        "RevSixSparseMAE": f(sparse["gaussian_retrieval_ensemble_mae"]),
        "RevSixSparseGain": f(sparse["gain_vs_canonical_prior"]),
        "RevSixSparseRelativeGain": f(
            100.0
            * sparse["gain_vs_canonical_prior"]
            / sparse["canonical_prior_trial_macro_mae"],
            2,
        ),
        "RevSixSparseGainPrevious": f(sparse["gain_vs_previous_ensemble"]),
        "RevSixSparseGainCI": ci(
            sparse["participant_video_crossed_bootstrap_ci95"]
        ),
        "RevSixSparseFoldGainMin": f(
            sparse_fold["gain_vs_canonical_prior_min"]
        ),
        "RevSixSparseFoldGainMean": f(
            sparse_fold["gain_vs_canonical_prior_mean"]
        ),
        "RevSixSparseFoldGainMax": f(
            sparse_fold["gain_vs_canonical_prior_max"]
        ),
        "RevSixSparsePositive": str(sparse_fold["positive_partitions"]),
        "RevSixSparseTotal": str(sparse_fold["total_partitions"]),
        "RevSixContentCandidates": str(content_config["candidate_count"]),
        "RevSixContentOuterSubjectFolds": str(content_config["subject_folds"]),
        "RevSixContentOuterVideoFolds": str(content_config["video_folds"]),
        "RevSixContentInnerSubjectFolds": str(
            content_config["inner_content_subject_folds"]
        ),
        "RevSixContentInnerVideoFolds": str(
            content_config["inner_content_video_folds"]
        ),
        "RevSixPhysioInnerSubjectFolds": str(
            physiology_config["inner_subject_folds"]
        ),
        "RevSixPhysioInnerVideoFolds": str(
            physiology_config["inner_video_folds"]
        ),
        "RevSixSparseInnerFolds": str(sparse_config["inner_folds"]),
        "RevSixSparseBootstrapRepeats": str(
            sparse_config["bootstrap_repeats"]
        ),
    }
    header = (
        [
            "% 由 scripts/generate_revision6_publication.py 自动生成。",
            "% 请勿编辑：paper_support/revision6_source.json 是唯一数值来源。",
        ]
        if chinese
        else [
            "% Generated by scripts/generate_revision6_publication.py.",
            "% Do not edit: paper_support/revision6_source.json is the sole numerical source.",
        ]
    )
    return "\n".join(header + [rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()]) + "\n"


def readme_block(source: dict[str, object], *, paper: bool) -> str:
    rev5 = source["summaries"]["revision5"]
    rev6 = source["summaries"]["revision6"]
    sensing = rev6["primary_antialias_reservation_sensing"]
    results = rows_by(source["tables"]["sensing_results"], "method")
    stack = results["context_plus_blockwise_residual"]
    repeat = sensing["repeated_grouped_cv_sensitivity"]
    gain = repeat["primary_route_gain_vs_video_mean_seconds"]
    signed = rev5["signed_calibration_practical_value"]["budgets"]
    replay = source["summaries"]["protocol_replay"]["validation"]
    external = source["summaries"]["protocol_replay"]["external_transfer"]
    algorithms = source["summaries"]["algorithm_experiments"]
    zero = algorithms[
        "zero_interaction_held_out_participant_held_out_released_video"
    ]
    physiology = algorithms["causal_physiology_residual"]["full_modalities"]
    sparse = algorithms["one_post_trial_sam_sparse_recovery"]
    stimulus = algorithms["stimulus_integrity"]
    title = "## Revision 6 唯一数值源与正式结论"
    location = "`paper_support/revision6_source.json`" if not paper else "`../paper_support/revision6_source.json`"
    consumers = "正文表格、结果宏、README 与正文图件" if paper else "汇总表格、结果摘要与图件"
    external_sources = (
        f"{external['source_count']} 篇独立作者团队的公开 HCI 论文"
        if paper else f"{external['source_count']} 项独立作者团队的公开 HCI 研究"
    )
    lines = [
        START,
        title,
        "",
        f"以下结果由 {location} 自动生成；{consumers}不得直接读取早期 revision 目录。",
        "",
        f"- 锁定协议已在 {replay['case_count']} 个路线级案例上完成结构重放，产生 {replay['distinct_audited_actions']} 种不同决策；"
        f"{replay['order_permutations']} 次顺序扰动结果不变，{replay['invalid_mutations_rejected']}/{replay['invalid_mutations']} 个非法记录被拒绝，"
        f"{replay['route_locality_probes_passed']}/{replay['route_locality_probes']} 个路线局部性探针、"
        f"{replay['permission_boundary_probes_passed']}/{replay['permission_boundary_probes']} 个正向授权边界探针及"
        f"{replay['precedence_probes_passed']}/{replay['precedence_probes']} 个声明优先级探针通过。"
        "这些检查只证明 schema、哈希、边型与传播规则的结构可复现性，不证明 substantive validity、分析者一致性或跨领域通用性。",
        f"- 同一声明式 resolver 还处理了 {external_sources}所形成的 {external['case_count']} 条路线，"
        f"无需 schema 扩展且没有 fallback；其中 {external['positive_bounded_actions']} 条被许可在已评估边界内个性化，"
        f"{external['constrained_or_study_actions']} 条保留比较方案或进入用户研究。五条规则逐条消融均改变至少一个动作。"
        "这是跨来源可表示性证据，不是独立分析者一致性或实质正确性证明。",
        f"- 原始刺激审计确认 {stimulus['files_present']}/{stimulus['files_expected']} 个文件存在且身份匹配。"
        f"在 held-out participant $\\times$ held-out released-video 的开发集 nested OOF 中，"
        f"元数据时间先验 MAE 为 {f(zero['metadata_prior_trial_macro_mae'])}，"
        f"冻结 CLIP+SigLIP 内容条件化先验为 {f(zero['selected_trial_macro_mae'])}（增益 {f(zero['selected_gain_vs_metadata'])}）。"
        "该结果只覆盖已发布的 15 个视频支持点，不是新视频总体主张。",
        f"- 在同一内容预测之上，主因果 EEG/fNIRS 残差门槛下 MAE 为 "
        f"{f(physiology['primary_trial_macro_mae'])}、增益 {f(physiology['primary_gain'])}，"
        "该受评价残差未证明增量，审计建议停用并保留内容。逐样本精确回退适用于残差停用后的输出；另一个时间令牌条件在主门槛下已选中零残差。不得把完整模态的受评价结果写成已等于基线，也不作等效或生理信号无信息解释。",
        f"- 每个试次结束后使用一个 SAM 效价--唤醒度对时，Gaussian retrieval + functional residual "
        f"稀疏恢复 MAE 为 {f(sparse['gaussian_retrieval_ensemble_mae'])}，"
        f"相对已知视频群体先验增益 {f(sparse['gain_vs_canonical_prior'])} "
        f"{ci(sparse['participant_video_crossed_bootstrap_ci95'])}。这是试后稀疏反馈，不是零交互实时预测。",
        f"- 严格抗混叠主分析中，video mean MAE 为 {f(results['video_mean']['primary_antialias_reservation_mae_seconds'])} s；"
        f"blockwise residual stack 为 {f(stack['primary_antialias_reservation_mae_seconds'])} s，"
        f"相对 video mean 的增益为 {f(stack['gain_vs_video_mean_seconds'])} s "
        f"{ci([stack['gain_vs_video_mean_ci95_low'], stack['gain_vs_video_mean_ci95_high']])}。结论是“所评估流程未证明增量”，不是等效性或生理信号无效。",
        f"- 五折 outer、四折 inner 的 grouped CV 共重复 {repeat['repeats']} 组参与者划分；每组均重建 reference 与 target。"
        f"主 sensing 路径增益范围为 {f(gain['minimum_across_partitions'])}--{f(gain['maximum_across_partitions'])} s，"
        f"{gain['partitions_positive']}/{gain['partitions_total']} 组划分为正。",
        f"- Signed calibration 的 1/2/4/8-video 增益为 "
        + "/".join(f(signed[str(b)]['calibrated_gain_vs_video_units']) for b in (1, 2, 4, 8))
        + " label units。阈值曲线只显示 24 名参与者的精确计数与比例，不再提供条件式 Wilson bands；"
        "它们不建立等效、实际价值或用户收益。",
        "- 原概化理论符号已移除，输出重命名为 relative/absolute shared-reference 描述性方差分数；重叠 leave-one-out 参考诱发的依赖未被常规 ANOVA 模型刻画，"
        "因此不再使用 .70 gate，也不作常规概化系数或可靠性解释。",
        "- 同试次 own-versus-donor 结果仅作为共享设备、共享时钟和共享任务条件下的界面依赖跨轴对齐证据；不把它解释为稳定个人属性。",
        "- 设计结论聚焦何时不应个性化：当增量、稳定性、后果终点或治理证据不足时，保留透明无传感器基线并记录 abstention。",
        END,
    ]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError("Malformed Revision-6 README markers")
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        return before.rstrip() + "\n\n" + block + after
    for legacy_heading in (
        "## Revision 5 正式结果",
        "## Revision 5 正式审查结论",
    ):
        if legacy_heading in text:
            before, tail = text.split(legacy_heading, 1)
            next_heading = tail.find("\n## ")
            after = "" if next_heading < 0 else tail[next_heading + 1 :]
            separator = "\n\n" if after else "\n"
            return before.rstrip() + "\n\n" + block + separator + after
    return text.rstrip() + "\n\n" + block + "\n"


def expected_files(source: dict[str, object]) -> dict[Path, str]:
    outputs = {
        GENERATED / "revision6_numbers.tex": macros(source),
        GENERATED / "revision6_numbers_zh.tex": macros(source, chinese=True),
        GENERATED / "revision6_signal_audit_rows.tex": signal_rows(source, chinese=False),
        GENERATED / "revision6_signal_audit_rows_zh.tex": signal_rows(source, chinese=True),
        GENERATED / "revision6_availability_rows.tex": availability_rows(source, chinese=False),
        GENERATED / "revision6_availability_rows_zh.tex": availability_rows(source, chinese=True),
        GENERATED / "revision6_sensing_rows.tex": sensing_rows(source, chinese=False),
        GENERATED / "revision6_sensing_rows_zh.tex": sensing_rows(source, chinese=True),
        GENERATED / "revision6_calibration_rows.tex": calibration_rows(source),
        GENERATED / "revision6_calibration_rows_zh.tex": calibration_rows(
            source, chinese=True
        ),
        GENERATED / "algorithm_route_rows.tex": algorithm_route_rows(
            source, chinese=False
        ),
        GENERATED / "algorithm_route_rows_zh.tex": algorithm_route_rows(
            source, chinese=True
        ),
        GENERATED / "algorithm_content_rows.tex": algorithm_content_rows(source),
        GENERATED / "algorithm_content_rows_zh.tex": algorithm_content_rows(
            source, chinese=True
        ),
        GENERATED / "algorithm_physiology_rows.tex": algorithm_physiology_rows(
            source
        ),
        GENERATED / "algorithm_physiology_rows_zh.tex": algorithm_physiology_rows(
            source, chinese=True
        ),
        GENERATED / "algorithm_sparse_rows.tex": algorithm_sparse_rows(source),
        GENERATED / "algorithm_sparse_rows_zh.tex": algorithm_sparse_rows(
            source, chinese=True
        ),
        GENERATED / "protocol_shortcut_rows.tex": protocol_shortcut_rows(source, chinese=False),
        GENERATED / "protocol_shortcut_rows_zh.tex": protocol_shortcut_rows(source, chinese=True),
        GENERATED / "protocol_replay_rows.tex": protocol_replay_rows(source, chinese=False),
        GENERATED / "protocol_replay_rows_zh.tex": protocol_replay_rows(source, chinese=True),
        GENERATED / "external_reuse_rows.tex": external_reuse_rows(source, chinese=False),
        GENERATED / "external_reuse_rows_zh.tex": external_reuse_rows(source, chinese=True),
        GENERATED / "protocol_rule_rows.tex": protocol_rule_rows(source, chinese=False),
        GENERATED / "protocol_rule_rows_zh.tex": protocol_rule_rows(source, chinese=True),
        GENERATED / "protocol_contract_rows.tex": protocol_contract_rows(
            source, chinese=False
        ),
        GENERATED / "protocol_contract_rows_zh.tex": protocol_contract_rows(
            source, chinese=True
        ),
        GENERATED / "adjacent_framework_rows.tex": adjacent_framework_rows(source),
        GENERATED / "adjacent_framework_rows_zh.tex": adjacent_framework_rows(
            source, chinese=True
        ),
        GENERATED / "statistical_analysis_registry_rows.tex": statistical_registry_rows(source),
        GENERATED / "statistical_analysis_registry_rows_zh.tex": statistical_registry_rows(
            source, chinese=True
        ),
        ROOT / "paper_support" / "statistical_analysis_registry.json": json.dumps(
            source["statistical_analysis_registry"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }
    for path, paper in ((ROOT / "README.zh-CN.md", False), (ROOT / "paper" / "README.md", True)):
        outputs[path] = replace_block(path.read_text(encoding="utf-8"), readme_block(source, paper=paper))
    return outputs


def main() -> None:
    args = parse_args()
    source = load_source()
    outputs = expected_files(source)
    stale = [path for path, expected in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != expected]
    if args.check:
        if stale:
            names = ", ".join(path.relative_to(ROOT).as_posix() for path in stale)
            raise SystemExit(f"Stale Revision-6 publication outputs: {names}")
        print("Revision-6 generated publication outputs are current")
        return
    for path, expected in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
