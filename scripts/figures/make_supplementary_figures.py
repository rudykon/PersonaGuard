"""Refresh SI source tables and figures; preserve the author-provided S1/S2 SVGs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath
import numpy as np
from PIL import Image
from vector_export import save_pdfua_embed
from manuscript_fonts import with_manuscript_fonts
import export_current_supplementary_svgs as exporter

AUTHOR_FIGURES = {
    "measurement_evidence_dag_full": "Evidence_Graph",
    "validation_information_boundaries": "Validation_Boundaries_portraits",
}

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "figures"
DATA_DIR = Path(__file__).resolve().parent
SOURCE = ROOT / "results/revision6_source.json"
WIDTH_MM = 182.1
DPI = 600
BLUE, ORANGE, PURPLE, GRAY = "#0072B2", "#D55E00", "#7A5195", "#53616D"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.facecolor": "white",
})

def load_source():
    source = json.loads(SOURCE.read_text())
    assert source["schema_version"] == "revision6-publication-source-v1"
    return source

def export(fig, stem):
    if stem in AUTHOR_FIGURES:
        plt.close(fig)
        exporter.export_one(AUTHOR_FIGURES[stem])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT / f"{stem}.tiff", dpi=DPI)
    save_pdfua_embed(fig, OUT / f"{stem}_embed.pdf", {})
    with Image.open(OUT / f"{stem}.png") as im:
        im.convert("L").save(OUT / f"{stem}_grayscale.png", dpi=(DPI, DPI))
    plt.close(fig)

def csv_rows(stem, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DATA_DIR / f"source_data_{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def card(ax, x, y, w, h, text, color=BLUE, fill="#F3F6F8", size=7):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=color,lw=.8,zorder=3))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,
            linespacing=1.35,zorder=4)
def arrow(ax, points, color=GRAY, style="-", lw=.9):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=8,
                                color=color,linestyle=style,lw=lw,zorder=2,clip_on=False))
def heading(ax, letter, title):
    ax.text(0,1.02,f"({letter})  {title}",weight="bold",size=8,va="bottom")
def blank(fig, bounds):
    ax=fig.add_axes(bounds)
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    return ax

EDGE_STYLE={
    "requires": (GRAY, "--", 1),
    "supports": (BLUE, "-", 1),
    "does_not_license": (ORANGE, "-.", 1),
    "blocks_deployment": (PURPLE, ":", 1.7),
}
# Geometry only. Source, target and relation are validated against the source graph.
NODES={
    "measurement_meaning": (.02,.79,"Measurement meaning","N1"),
    "estimator_robustness": (.02,.60,"Estimator robustness","N2"),
    "empirical_library_resampling_stability": (.02,.41,"Empirical-library\nresampling stability","N3"),
    "trace_derived_estimation": (.385,.79,"Trace-derived\nestimation","N4"),
    "absolute_q_calibration": (.385,.60,"Absolute-q calibration","N5"),
    "eeg_fnirs_increment": (.385,.41,"EEG/fNIRS increment","N6"),
    "signed_proximal_correction": (.385,.22,"Signed proximal\ncorrection","N7"),
    "downstream_user_utility": (.765,.79,"Downstream user utility","N8"),
    "session_persistence": (.765,.60,"Session persistence","N9"),
    "reference_representativeness_equity": (.765,.41,"Reference representation\nand equity","N10"),
    "retention_and_transfer": (.765,.22,"Retention and transfer","N11"),
}
ROUTES={
 ("measurement_meaning","trace_derived_estimation"): [( .245,.85),(.385,.85)],
 ("measurement_meaning","absolute_q_calibration"): [(.245,.82),(.29,.82),(.29,.66),(.385,.66)],
 ("measurement_meaning","signed_proximal_correction"): [(.245,.80),(.315,.80),(.315,.28),(.385,.28)],
 ("estimator_robustness","empirical_library_resampling_stability"): [(.1325,.60),(.1325,.53)],
 ("empirical_library_resampling_stability","absolute_q_calibration"): [(.245,.50),(.275,.50),(.275,.63),(.385,.63)],
 ("empirical_library_resampling_stability","eeg_fnirs_increment"): [(.245,.46),(.385,.46)],
 ("trace_derived_estimation","signed_proximal_correction"): [(.61,.82),(.65,.82),(.65,.25),(.61,.25)],
 ("eeg_fnirs_increment","trace_derived_estimation"): [(.61,.50),(.635,.50),(.635,.87),(.61,.87)],
 ("signed_proximal_correction","downstream_user_utility"): [(.61,.30),(.685,.30),(.685,.86),(.765,.86)],
 ("absolute_q_calibration","downstream_user_utility"): [(.61,.67),(.71,.67),(.71,.82),(.765,.82)],
 ("reference_representativeness_equity","retention_and_transfer"): [(.88,.41),(.88,.34)],
 ("session_persistence","retention_and_transfer"): [(.99,.66),(.999,.66),(.999,.28),(.99,.28)],
 ("downstream_user_utility","retention_and_transfer"): [(.99,.85),(.999,.85),(.999,.935),(.735,.935),(.735,.28),(.765,.28)],
 ("signed_proximal_correction","retention_and_transfer"): [(.61,.23),(.61,.18),(.81,.18),(.81,.22)],
}

def figure_graph(source):
    graph=source["summaries"]["evidence_traceability"]
    nodes={n["id"]: n for n in graph["nodes"]}
    assert set(nodes)==set(NODES)
    assert {(e["source"],e["target"]) for e in graph["edges"]}==set(ROUTES)
    assert len(nodes)==11 and len(graph["edges"])==14
    fig=plt.figure(figsize=(WIDTH_MM/25.4,7.7))
    ax=blank(fig,[.02,.31,.96,.64])
    heading(ax,"A","Complete worked-case evidence graph")
    for x,title in [( .13,"Measurement and\nconstruction"),(.50,"Estimation, sensing\nand calibration"),(.88,"Consequences\nand reuse")]:
        ax.text(x,.985,title,ha="center",va="top",size=7,weight="bold")
    for edge in graph["edges"]:
        color,style,lw=EDGE_STYLE[edge["type"]]
        arrow(ax,ROUTES[(edge["source"],edge["target"])],color,style,lw)
    status_colors={"AVAILABLE":BLUE,"PARTIAL":ORANGE,"NOT_TESTED":GRAY,
                   "NO_DEMONSTRATED_INCREMENT":GRAY,"NOT_JUSTIFIED":PURPLE}
    for key,(x,y,title,short) in NODES.items():
        state=nodes[key]["status"]
        c=status_colors[state]
        card(ax,x,y,.225,.12,"",c)
        ax.text(x+.012,y+.111,short,size=6.2,color=GRAY,va="top",zorder=4)
        ax.text(x+.1125,y+.065,title,size=6.6,ha="center",va="center",weight="bold",zorder=4)
        label=state if state!="NO_DEMONSTRATED_INCREMENT" else "NO_DEMONSTRATED\n_INCREMENT"
        ax.text(x+.1125,y+.021,label,size=6.1,ha="center",va="center",color=c,zorder=4)
    handles=[mpl.lines.Line2D([],[],color=c,ls=st,lw=lw,label=k)
             for k,(c,st,lw) in EDGE_STYLE.items()]
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.082),
              ncol=2,fontsize=6.5,handlelength=3,columnspacing=2)
    ax.text(.5,.032,"Evidence states are node-local; edges do not automatically propagate permission.",
            ha="center",size=6.6)
    ax.text(.5,.005,"11 evidence nodes are not 11 resolved audit routes.",ha="center",size=6.6)
    b=blank(fig,[.03,.035,.94,.23])
    heading(b,"B","Trace an action to its source")
    case=next(c for c in source["summaries"]["protocol_replay_cases"]["cases"] if c["id"]=="signed_calibration")
    result=next(r for r in source["summaries"]["protocol_replay"]["worked_case_results"] if r["id"]=="signed_calibration")
    assert result["matched_rule_id"] == "R3_CONSEQUENCE_MATCH"
    ev=case["evidence"]
    card(b,.00,.43,.26,.46,"Source locator\n"+case["source_locator"].replace("results/evidence_traceability.json#","evidence_traceability.json\n#"),size=6.4)
    card(b,.30,.43,.32,.46,f"Comparator: {ev['comparator_increment']}\nEvaluation: {ev['evaluation_mode']}\nOutcome: {ev['user_outcome']}",size=6.7)
    card(b,.67,.43,.33,.46,"R3 matches before R1\nIn-context user study\nbefore deployment",PURPLE,size=7)
    arrow(b,[(.26,.66),(.30,.66)])
    arrow(b,[(.62,.66),(.67,.66)])
    card(b,.30,.02,.70,.27,"Re-review: obtain consequence-matched evidence;\nupdate the record and rerun all rules, not automatic permission.",GRAY,size=6.7)
    arrow(b,[(.835,.43),(.835,.29)])
    b.text(.01,.17,"Existing record;\nno additional case.",size=6.5,va="center")
    rows=[{"panel":"A","kind":"node","node":n["id"],"display_id":NODES[n["id"]][3],
           "status":n["status"],"estimand":n["estimand"]} for n in graph["nodes"]]
    rows += [{"panel":"A","kind":"edge",**e} for e in graph["edges"]]
    rows += [{"panel":"B","kind":"walkthrough","case_id":case["id"],"source_locator":case["source_locator"],
              "comparator_increment":ev["comparator_increment"],"evaluation_mode":ev["evaluation_mode"],
              "user_outcome":ev["user_outcome"],"action":result["audited_action"],
              "review_trigger":nodes["signed_proximal_correction"]["review_trigger"]}]
    csv_rows("measurement_evidence_dag_full",rows)
    export(fig,"measurement_evidence_dag_full")
    return rows

def boundary_records(source):
    s=source["summaries"]
    r=s["revision6"]["primary_antialias_reservation_sensing"]["repeated_grouped_cv_sensitivity"]
    c=s["av_content_run_manifest"]["configuration"]
    p=s["av_physio_run_manifest"]["configuration"]
    sam=s["sparse_anchor_run_manifest"]["configuration"]
    return [
      dict(panel="A",condition="scalar_profile_sensing",outer_participant_folds=r["outer_folds_per_repeat"],inner_participant_folds=r["inner_folds"],prediction_input="complete-trial physiology; no target-video joystick",evaluation_target="q constructed against training reference",source_locator="summaries.revision6.primary_antialias_reservation_sensing"),
      dict(panel="A",condition="signed_calibration",calibration_budgets=";".join(source["summaries"]["revision5"]["signed_calibration_practical_value"]["budgets"]),prediction_input="other calibration-video reports",evaluation_target="separate evaluation-video reference-proximal error",source_locator="summaries.revision5.signed_calibration_practical_value"),
      dict(panel="B",condition="dense_content",outer_participant_folds=c["subject_folds"],outer_video_folds=c["video_folds"],inner_participant_folds=c["inner_content_subject_folds"],inner_video_folds=c["inner_content_video_folds"],prediction_input="complete target-video asset; no target-video fitting labels",source_locator="summaries.av_content_run_manifest.configuration"),
      dict(panel="B",condition="dense_causal_residual",inner_participant_folds=p["inner_subject_folds"],inner_video_folds=p["inner_video_folds"],prediction_input="frozen content and current/past physiology",source_locator="summaries.av_physio_run_manifest.configuration"),
      dict(panel="C",condition="known_video_sam",inner_participant_folds=sam["inner_folds"],prediction_input="same-video training trajectories and held-out post-trial SAM",evaluation_target="held-out dense report; evaluation only",source_locator="summaries.sparse_anchor_run_manifest.configuration"),
      dict(panel="D",condition="applicable_training_boundary",prediction_input="outer errors do not choose inner candidates or gates",source_locator="statistical_analysis_registry"),
    ]

def figure_boundaries(source):
    records=boundary_records(source); a,c,p,sam=records[0],records[2],records[3],records[4]
    budgets = records[1]["calibration_budgets"].replace(";", " / ")
    fig=plt.figure(figsize=(WIDTH_MM/25.4,7.4))
    axes=[blank(fig,b) for b in [[.035,.535,.43,.41],[.555,.535,.41,.41],[.035,.07,.43,.40],[.555,.07,.41,.40]]]
    ax=axes[0]; heading(ax,"A","Participant-grouped profile analyses")
    ax.text(.5,.96,f"{a['outer_participant_folds']} outer × {a['inner_participant_folds']} inner participant folds",ha="center",size=7)
    card(ax,0,.73,.45,.17,"Outer training\nInner selection")
    card(ax,.55,.73,.45,.17,"Held-out participant",ORANGE)
    arrow(ax,[(.45,.80),(.55,.80)])
    card(ax,0,.48,1,.18,"Scalar sensing: full-trial physiology → q prediction\nTarget-video report → evaluation target only",size=6.7)
    arrow(ax,[(.77,.73),(.77,.66)])
    card(ax,0,.21,1,.20,f"Signed calibration within held-out participant\n{budgets} calibration videos → correction\nSeparate evaluation videos → error",size=6.6)
    arrow(ax,[(.22,.73),(.22,.66)])
    arrow(ax,[(.99,.80),(1.03,.80),(1.03,.31),(1,.31)])
    ax.text(.5,.105,"No target-video joystick input for scalar sensing.",ha="center",size=6.5,weight="bold")
    ax.text(.5,.025,"Calibration reports are permitted inputs only\nfor the declared calibration condition.",ha="center",size=6.5)
    ax=axes[1]; heading(ax,"B","Dense participant × video holdout")
    ax.text(.5,.96,f"{c['outer_participant_folds']} × {c['outer_video_folds']} outer participant/video cells",ha="center",size=7)
    ax.text(.47,.87,"Training videos",ha="center",size=6.3); ax.text(.82,.87,"Held-out videos",ha="center",size=6.3)
    ax.text(.13,.75,"Training\nparticipants",ha="center",va="center",size=6.3)
    ax.text(.13,.58,"Held-out\nparticipants",ha="center",va="center",size=6.3)
    for x,y,text_,fill in [( .30,.67,"Fitting labels\navailable","#DFEDF6"),(.65,.67,"Not used\nfor fitting","#F3F6F8"),(.30,.50,"Not used\nfor fitting","#F3F6F8"),(.65,.50,"Outer evaluation\ntargets","#FBEFE4")]:
        card(ax,x,y,.34,.16,text_,fill=fill,size=6.4)
    ax.text(.5,.425,f"Inner cells: content {c['inner_participant_folds']} × {c['inner_video_folds']}; residual {p['inner_participant_folds']} × {p['inner_video_folds']}",ha="center",size=6.6)
    card(ax,0,.235,1,.14,"Target-video asset → content predictor\nCurrent/past physiology → residual",size=6.6)
    ax.text(.5,.155,"Physiology:  t−2 — t−1 — t  |  t+1 …",ha="center",size=6.8)
    ax.text(.5,.10,"permitted             excluded",ha="center",size=6.5)
    ax.text(.5,.015,"Causal timeline applies to the residual only,\nnot offline full-trial scalar sensing.",ha="center",size=6.4)
    ax=axes[2]; heading(ax,"C","Known-video, post-trial SAM")
    card(ax,0,.76,1,.16,"Training participants' trajectories for video v\n→ same-video library",size=6.8)
    card(ax,0,.53,1,.16,"Held-out participant's post-trial SAM pair\n→ retrieval and recovery",PURPLE,size=6.8)
    arrow(ax,[(.5,.76),(.5,.69)])
    card(ax,0,.32,1,.14,"Held-out dense report → evaluation only",GRAY,size=6.6)
    ax.text(.5,.23,f"Participant holdout; {sam['inner_participant_folds']} inner folds. Known video.",ha="center",size=6.6)
    ax.text(.5,.145,"Playback → trial end → SAM → reconstruction",ha="center",size=6.5)
    ax.text(.5,.035,"New participant, not unknown-video prediction.\nPost-trial reconstruction, not online personalization.",ha="center",size=6.4)
    ax=axes[3]; heading(ax,"D","Fold-local construction and selection")
    ax.add_patch(Rectangle((.005,.465),.99,.48,fill=False,ls="--",ec=GRAY,lw=.7))
    labels=["Declare outer split","Rebuild training references / targets / scaling","Inner candidate and gate selection","Produce outer-held-out predictions","Compute route-matched errors","Aggregate and report"]
    ys=[.80,.64,.48,.32,.16,0]
    for y,label in zip(ys,labels):
        card(ax,.04,y,.92,.12,label,size=6.4)
    for y in ys[:-1]:
        arrow(ax,[(.5,y),(.5,y-.04)])
    ax.text(.5,.966,"Within each applicable training boundary",ha="center",size=6.4)
    ax.text(1.03,.35,"No outer-error feedback",rotation=90,ha="center",va="center",size=6.3,color=ORANGE)
    arrow(ax,[(.96,.22),(.99,.22),(.99,.54),(.96,.54)],ORANGE,"--",.8)
    ax.text(.99,.39,"×",color=ORANGE,size=11,ha="center",va="center",zorder=5)
    fig.text(.5,.02,"Declared information flow and validation assumptions—not an independent leakage audit.",ha="center",size=6.5,color=GRAY)
    csv_rows("validation_information_boundaries",records)
    export(fig,"validation_information_boundaries")

def algorithm_rows(source):
    a=source["summaries"]["algorithm_experiments"]
    content=a["zero_interaction_held_out_participant_held_out_released_video"]
    rows=[]
    for c in content["candidates"]:
        rows.append(dict(panel="A",condition=c["name"],estimate=c["trial_macro_mae"],
                         low=c["offset_mae_range"][0],high=c["offset_mae_range"][1],
                         interval="fixed-offset minimum--maximum; not CI",
                         comparator="metadata prior",comparator_mae=content["metadata_prior_trial_macro_mae"],
                         retained=c["name"]==content["selected_model"]))
    for family in ("full_modalities","temporal_token_modalities"):
        f=a["causal_physiology_residual"][family]
        for r in f["decision_curve"]:
            rows.append(dict(panel="B",condition=family,threshold=r["minimum_inner_gain"],
                             estimate=r["gain_vs_content_prior"],mae=r["trial_macro_mae"],
                             primary=r["minimum_inner_gain"]==f["primary_minimum_inner_gain"],
                             comparator="frozen content prior",interval="none"))
    sparse=a["one_post_trial_sam_sparse_recovery"]; base=sparse["canonical_prior_trial_macro_mae"]
    for key,mae in [("canonical_prior",base),("previous_functional_ensemble",sparse["previous_functional_ensemble_mae"]),
                    ("gaussian_functional_ensemble",sparse["gaussian_retrieval_ensemble_mae"])]:
        row=dict(panel="C",condition=key,estimate=base-mae,mae=mae,comparator="known-video population prior",comparator_mae=base,interval="none")
        if key=="gaussian_functional_ensemble":
            row["estimate"] = sparse["gain_vs_canonical_prior"]
            row.update(low=sparse["participant_video_crossed_bootstrap_ci95"][0],
                       high=sparse["participant_video_crossed_bootstrap_ci95"][1],
                       interval="participant x video crossed-bootstrap 95% percentile CI")
        rows.append(row)
    return rows

@with_manuscript_fonts
def figure_algorithms(source):
    rows=algorithm_rows(source); fig=plt.figure(figsize=(WIDTH_MM/25.4,7.2))
    ax=fig.add_axes([.32,.60,.65,.32])
    content=[r for r in rows if r["panel"]=="A"]
    names=["CLIP","SigLIP","DINOv2","Equal-norm CLIP + SigLIP","Weighted CLIP + SigLIP","+ 1-s feature difference","+ 1- and 3-s differences"]
    for y,r in enumerate(content):
        color=BLUE if r["retained"] else GRAY
        ax.plot([r["low"],r["high"]],[y,y],color=color,lw=1.5)
        ax.plot(r["estimate"],y,"o",color=color,ms=4)
        if r["retained"]: ax.plot(r["estimate"],y,"o",mfc="none",mec=BLUE,ms=8)
    ax.axvline(content[0]["comparator_mae"],ls="--",color=GRAY,lw=.8)
    ax.set(yticks=range(len(names)),yticklabels=names,ylim=(6.6,-.7),xlim=(30.4,31.39),
           xlabel="Trial-macro MAE (1–255 scale) — lower is better")
    ax.set_title("(A)  Content representations and fixed-offset sensitivity",loc="left",pad=16,weight="bold")
    ax.text(.98,1.01,"Metadata prior",transform=ax.transAxes,ha="right",size=6.4,color=GRAY)
    ax.text(0,-.28,"Segments: fixed −3 to +3 s min–max, not confidence intervals.\nOpen ring: retained representation; primary estimates use the frozen origin.",transform=ax.transAxes,size=6.4)
    ax=fig.add_axes([.10,.145,.37,.275])
    for family,label,color,marker in [("full_modalities","Full causal modalities",BLUE,"o"),("temporal_token_modalities","CBraMod temporal-token views",ORANGE,"s")]:
        data=[r for r in rows if r["panel"]=="B" and r["condition"]==family]
        ax.plot([r["threshold"] for r in data],[r["estimate"] for r in data],
                label=label,c=color,marker=marker,ms=4,lw=1,ls="-" if marker=="o" else "--")
    primary = next(r["threshold"] for r in rows if r["panel"]=="B" and r["primary"])
    ax.axhline(0,c=GRAY,lw=.8); ax.axvline(primary,c=PURPLE,ls=":",lw=1)
    ax.set(xlabel="Minimum inner-selection gain",ylabel="Outer MAE gain over content prior",
           xticks=[0,.05,.1,.2,.5],ylim=(-.009,.048))
    ax.set_title("(B)  Residual selection gates",loc="left",pad=12,weight="bold")
    ax.legend(loc="upper right",fontsize=6.1)
    ax.text(.32,.021,f"Primary gate: {primary:.2f}\nDevelopment, not preregistered",size=6.1,color=PURPLE,ha="center")
    ax=fig.add_axes([.70,.145,.27,.275])
    recovery=[r for r in rows if r["panel"]=="C"]
    for y,r in enumerate(recovery):
        if "low" in r: ax.plot([r["low"],r["high"]],[y,y],c=PURPLE,lw=1)
        ax.plot(r["estimate"],y,"o",c=PURPLE if y==2 else GRAY,ms=4)
    ax.axvline(0,c=GRAY,lw=.8)
    ax.set(yticks=[0,1,2],yticklabels=["Canonical\npopulation prior","Previous functional\nensemble","Gaussian retrieval\n+ functional residual"],
           ylim=(2.6,-.6),xlabel="MAE gain over known-video\npopulation prior",xlim=(-.2,4.1))
    ax.set_title("(C)  Post-trial recovery",loc="left",pad=12,weight="bold")
    ax.text(-.35,-.36,"Final ensemble: crossed-bootstrap 95% CI.\nOther conditions: point estimates only.",transform=ax.transAxes,size=6.2)
    fig.text(.5,.015,"Different comparators and information conditions; no common ranking. Development evidence only.",ha="center",size=6.5,color=GRAY)
    csv_rows("dense_algorithm_sensitivity",rows)
    export(fig,"dense_algorithm_sensitivity")

def main():
    source=load_source()
    figure_graph(source); figure_boundaries(source); figure_algorithms(source)
    print("Refreshed SI source tables; exported author S1/S2 and canonical S3.")
if __name__=="__main__":
    main()
