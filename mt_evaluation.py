# %%
import os

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.metrics import accuracy_score, precision_score, recall_score

from mt import z_score

traj_mrs = ("resize", "rotate", "flip", "flipud", "fliplr")
map_mrs = ("classchange", "obstacle")

# %%
print("Load results")
dfs = []

df1 = pd.read_csv("results_sdd_trajnet_full.csv")
df1["setting"] = "sdd_trajnet"
dfs.append(df1)

df2 = pd.read_csv("results_sdd_longterm_full.csv")
df2["setting"] = "sdd_longterm"
dfs.append(df2)

if os.path.exists("results_ind_longterm_full.csv"):
    df3 = pd.read_csv("results_ind_longterm_full.csv")
    df3["setting"] = "ind_longterm"
    dfs.append(df3)
else:
    print("No results_ind_longterm_full.csv found, skipping inD long-term evaluation.")

df = pd.concat(dfs)
df["mrmain"] = df.mr.str.split("_").str[0]

df.loc[df.mr == "obstacle", "change_type"] = "ClassChangeType.OBSTACLE"

dft = pd.DataFrame(df[df.mrmain.isin(traj_mrs)])
dfm = pd.DataFrame(df[df.mrmain.isin(map_mrs)])

df.sample(10)
# %%
dft[
    [
        "source_bonfde_mean",
        "source_bonade_mean",
        "source_meanfde_mean",
        "source_meanade_mean",
    ]
] = (
    dft[
        [
            "setting",
            "scene",
            "mr",
            "input_traj_id",
            "source_fde",
            "source_ade",
            "source_fde_mean",
            "source_ade_mean",
        ]
    ]
    .groupby(["setting", "scene", "mr", "input_traj_id"])
    .transform(np.mean)
)
dft[
    [
        "source_bonfde_std",
        "source_bonade_std",
        "source_meanfde_std",
        "source_meanade_std",
    ]
] = (
    dft[
        [
            "setting",
            "scene",
            "mr",
            "input_traj_id",
            "source_fde",
            "source_ade",
            "source_fde_mean",
            "source_ade_mean",
        ]
    ]
    .groupby(["setting", "scene", "mr", "input_traj_id"])
    .transform(np.std)
)

# %%
print("Calculate scores and p-values")

dft[["zscore_bonade", "pvalue_bonade"]] = dft.apply(
    lambda d: pd.Series(
        z_score(d.follow_ade, d.source_bonade_mean, d.source_bonade_std)
    ),
    axis=1,
)
dft[["zscore_bonfde", "pvalue_bonfde"]] = dft.apply(
    lambda d: pd.Series(
        z_score(d.follow_fde, d.source_bonfde_mean, d.source_bonfde_std)
    ),
    axis=1,
)
dft[["zscore_meanade", "pvalue_meanade"]] = dft.apply(
    lambda d: pd.Series(
        z_score(d.follow_ade_mean, d.source_meanade_mean, d.source_meanade_std)
    ),
    axis=1,
)
dft[["zscore_meanfde", "pvalue_meanfde"]] = dft.apply(
    lambda d: pd.Series(
        z_score(d.follow_fde_mean, d.source_meanfde_mean, d.source_meanfde_std)
    ),
    axis=1,
)


# %%
MR_NAMES = {
    "flipud": "Mirror-v",
    "fliplr": "Mirror-h",
    "rotate_90": "Rotate-90",
    "rotate_180": "Rotate-180",
    "rotate_270": "Rotate-270",
    "resize_02": "Resize-0.2",
    "resize_03": "Resize-0.3",
    "classchange": "Class Change",
    "obstacle": "Obstacle",
}
SETTING_NAMES = {
    "sdd_trajnet": "SDD (Short)",
    "sdd_longterm": "SDD (Long)",
    "ind_longterm": "inD (Long)",
}
COLUMN_NAMES_TRAJ = {
    "mr": "MR",
    "setting": "D",
    "pvalue": "WVC",
    "pvalue_bonade": "B-ADE",
    "pvalue_bonfde": "B-FDE",
    "pvalue_meanade": "M-ADE",
    "pvalue_meanfde": "M-FDE",
    "waypoint_map_hellinger": "HVC",
}

hvc = (
    dft[["setting", "mr", "waypoint_map_hellinger"]]
    .groupby(["setting", "mr"])
    .agg(["mean", "std"])
    .round(2)
    .apply(lambda x: f"{x.iloc[0]:.2f}±{x.iloc[1]:.2f}", axis=1)
)


results = (
    dft[
        [
            "setting",
            "mr",
            "pvalue",
            "pvalue_bonade",
            "pvalue_bonfde",
            "pvalue_meanade",
            "pvalue_meanfde",
        ]
    ]
    .groupby(["setting", "mr"])
    .agg(lambda s: np.mean(s <= 0.05))
)
results["waypoint_map_hellinger"] = hvc
results = results.reset_index()
results["mr"] = results.mr.map(MR_NAMES)
results["setting"] = results.setting.map(SETTING_NAMES)
# Sort by setting and then by the order defined in MR_NAMES
results["mr"] = pd.Categorical(
    results["mr"], categories=list(MR_NAMES.values()), ordered=True
)
results["setting"] = pd.Categorical(
    results["setting"], categories=list(SETTING_NAMES.values()), ordered=True
)
results = results.sort_values(["setting", "mr"])
# Rename columns using METRIC_NAMES
results = results.rename(columns=COLUMN_NAMES_TRAJ)
latex_table = results.to_latex(
    float_format="{:.1%}".format, escape=True, index=False, column_format="ll|ccccc|c"
).replace("\%", "")
latex_table = latex_table.replace(
    f"{SETTING_NAMES['sdd_trajnet']}",
    "\\parbox[t]{2mm}{\\multirow{7}{*}{\\rotatebox[origin=c]{90}{SDD (Short)}}}",
    1,
).replace(f"{SETTING_NAMES['sdd_trajnet']} &", " &")
latex_table = latex_table.replace(
    f"{SETTING_NAMES['sdd_longterm']}",
    "\\midrule\n\\parbox[t]{2mm}{\\multirow{7}{*}{\\rotatebox[origin=c]{90}{SDD (Long)}}}",
    1,
).replace(f"{SETTING_NAMES['sdd_longterm']} &", " &")
latex_table = latex_table.replace(
    f"{SETTING_NAMES['ind_longterm']}",
    "\\midrule\n\\parbox[t]{2mm}{\\multirow{7}{*}{\\rotatebox[origin=c]{90}{inD (Long)}}}",
    1,
).replace(f"{SETTING_NAMES['ind_longterm']} &", " &")

latex_table = latex_table.replace(
    "  &   Rotate-90", "\\cmidrule(lr){2-8}\n  &   Rotate-90"
)
latex_table = latex_table.replace(
    "  &  Resize-0.2", "\\cmidrule(lr){2-8}\n  &  Resize-0.2"
)
print("Results for label-preserving MRs (written to results_trajectory.tex):")
print(latex_table)
open("results_trajectory.tex", "w").write(latex_table)
results

# %%
# Table for map-based MRs

COLUMN_NAMES_MAP = {
    "mr": "MR",
    "setting": "D",
    "pvalue": "HTC",
    "intersections": "Intersections",
    "waypoint_map_hellinger": "HVC",
    "change_type": "Effect",
}
CLASS_CHANGE_MAP = {
    "ClassChangeType.MORE_WALKABLE": "Increase",
    "ClassChangeType.LESS_WALKABLE": "Decrease",
    "ClassChangeType.OBSTACLE": "Obstacle",
}

results_map = (
    dfm[
        [
            "setting",
            "mrmain",
            "change_type",
            "intersections",
            "pvalue",
        ]
    ]
    .groupby(["setting", "mrmain", "change_type"])
    .agg(
        {
            "intersections": lambda s: np.mean(s),
            "pvalue": lambda s: np.mean(s <= 0.05),
        }
    )
)

hvc = (
    dfm[["setting", "mrmain", "change_type", "waypoint_map_hellinger"]]
    .groupby(["setting", "mrmain", "change_type"])
    .agg(["mean", "std"])
    .round(2)
    .apply(lambda x: f"{x.iloc[0]:.2f}±{x.iloc[1]:.2f}", axis=1)
)
results_map["waypoint_map_hellinger"] = hvc

results_map = results_map.reset_index()

results_map["mr"] = results_map.mrmain.map(MR_NAMES)
del results_map["mrmain"]
results_map["setting"] = results_map.setting.map(SETTING_NAMES)
results_map["change_type"] = results_map.change_type.map(CLASS_CHANGE_MAP)
# Sort by setting and then by the order defined in MR_NAMES
results_map["mr"] = pd.Categorical(
    results_map["mr"], categories=list(MR_NAMES.values()), ordered=True
)
results_map["setting"] = pd.Categorical(
    results_map["setting"], categories=list(SETTING_NAMES.values()), ordered=True
)
results_map["change_type"] = pd.Categorical(
    results_map["change_type"], categories=list(CLASS_CHANGE_MAP.values()), ordered=True
)
results_map = results_map.sort_values(["setting", "mr", "change_type"])
results_map = results_map.rename(columns=COLUMN_NAMES_MAP)
results_map = results_map[["D", "MR", "Effect", "HTC", "Intersections", "HVC"]]

latex_table = results_map.to_latex(
    float_format="{:.1%}".format,
    escape=True,
    index=False,
    column_format="lll|rrr",
    na_rep="--",
).replace("\%", "")
latex_table = latex_table.replace(
    f"{SETTING_NAMES['sdd_trajnet']}",
    "\\parbox[t]{2mm}{\\multirow{4}{*}{\\rotatebox[origin=c]{90}{SDD (Short)}}}",
    1,
).replace(f"{SETTING_NAMES['sdd_trajnet']} &", " &")
latex_table = latex_table.replace(
    f"{SETTING_NAMES['sdd_longterm']}",
    "\\midrule\n\\parbox[t]{2mm}{\\multirow{4}{*}{\\rotatebox[origin=c]{90}{SDD (Long)}}}",
    1,
).replace(f"{SETTING_NAMES['sdd_longterm']} &", " &")
latex_table = latex_table.replace(
    f"{SETTING_NAMES['ind_longterm']}",
    "\\midrule\n\\parbox[t]{2mm}{\\multirow{4}{*}{\\rotatebox[origin=c]{90}{inD (Long)}}}",
    1,
).replace(f"{SETTING_NAMES['ind_longterm']} &", " &")
latex_table = latex_table.replace("nan", "--")

print("Results for map-based MRs (written to results_map.tex):")
print(latex_table)
open("results_map.tex", "w").write(latex_table)
results_map

# %%
print("Calculate scores for p-value thresholds")
scores = []

for thresh in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2):
    for setting in dft["setting"].unique():
        cdf_setting = (
            dft[dft["setting"] == setting][
                [
                    "pvalue",
                    "pvalue_bonade",
                    "pvalue_bonfde",
                    "pvalue_meanade",
                    "pvalue_meanfde",
                ]
            ]
            <= thresh
        )

        for t in ("pvalue_bonade", "pvalue_bonfde", "pvalue_meanade", "pvalue_meanfde"):
            scores.append(
                (
                    setting,
                    t,
                    thresh,
                    "precision",
                    precision_score(cdf_setting["pvalue"], cdf_setting[t]),
                )
            )
            scores.append(
                (
                    setting,
                    t,
                    thresh,
                    "recall",
                    recall_score(cdf_setting["pvalue"], cdf_setting[t]),
                )
            )
            scores.append(
                (
                    setting,
                    t,
                    thresh,
                    "accuracy",
                    accuracy_score(cdf_setting["pvalue"], cdf_setting[t]),
                )
            )

sdf = pd.DataFrame.from_records(
    scores, columns=["Setting", "Criterion", "threshold", "Metric", "score"]
)
sdf["score"] *= 100
sdf["Metric"] = sdf.Metric.str.title()
sdf["Setting"] = sdf["Setting"].map(SETTING_NAMES)
sdf["Setting"] = pd.Categorical(
    sdf["Setting"], categories=list(SETTING_NAMES.values()), ordered=True
)
sdf.sort_values(["Setting", "Criterion", "threshold", "Metric"], inplace=True)

plot = (
    p9.ggplot(
        data=sdf.replace(
            {
                "pvalue_bonade": "BoN-ADE",
                "pvalue_bonfde": "BoN-FDE",
                "pvalue_meanade": "Mean-ADE",
                "pvalue_meanfde": "Mean-FDE",
            }
        ),
        mapping=p9.aes(x="threshold", y="score", linetype="Metric", color="Criterion"),
    )
    + p9.facet_wrap("~ Setting", ncol=3)
    + p9.geom_line(size=1.2)
    + p9.xlab("p-value Threshold")
    + p9.ylab("Scores (%)")
    + p9.ylim(20, 100)
    + p9.scale_x_continuous(
        breaks=(0.01, 0.05, 0.1, 0.15, 0.2),
        labels=["0.01", "0.05", "0.1", "0.15", "0.2"],
    )
    + p9.scale_linetype_manual(
        values={
            "Accuracy": "solid",
            "Precision": "dashed",
            "Recall": "dotted",
        }
    )
    + p9.theme_minimal()
    + p9.theme(
        text=p9.element_text(size=12),
        axis_title=p9.element_text(size=14, weight="bold"),
        axis_text=p9.element_text(size=12),
        legend_title=p9.element_text(size=12, weight="bold"),
        legend_text=p9.element_text(size=10),
        legend_position="top",
        legend_box="horizontal",
        figure_size=(12, 4),
    )
)

# Save the plot
plot.save("scores_trajectory.pdf", dpi=400)
print("Plot saved as scores_trajectory.pdf")

plot
# %%
