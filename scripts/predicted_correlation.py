# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "pandas>=2",
#     "scikit-learn>=1.4",
#     "seaborn>=0.13",
# ]
# ///
"""Plot feature-importance and objective-correlation figures from predictions.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.csv"
df = pd.read_csv(DATA_PATH)
feature_cols = ["Mo", "Nb", "Ta", "W", "Co", "Hf"]
targets = ["Printability", "Yield Strength", "Crack Susceptibility", "Lattice Misfit", "Young Modulus", "Ductility", "Risk"]

all_importances = pd.DataFrame(index=feature_cols)
X = df[feature_cols].values
for target in targets:
    y = df[target].values
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    all_importances[target] = rf.feature_importances_

all_importances.loc["CoHf"] = all_importances.loc["Co"] + all_importances.loc["Hf"]
features_reduced = ["Mo", "Nb", "Ta", "W", "CoHf"]
all_importances_reduced = all_importances.loc[features_reduced]

color_map = {
    "Mo": "#e39e34",
    "Nb": "#ac193d",
    "Ta": "#f7e66c",
    "W": "#3e934b",
    "CoHf": "#9ed18e",
}

fig, ax = plt.subplots(figsize=(8, 6))
left = np.zeros(len(targets))
for feature in features_reduced:
    ax.barh(
        targets,
        all_importances_reduced.loc[feature],
        left=left,
        label=feature,
        color=color_map[feature],
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )
    left += all_importances_reduced.loc[feature].values

ax.set_xlabel("Feature Importance", fontsize=14, labelpad=10)
ax.set_ylabel("Target Property", fontsize=14, labelpad=10)
ax.set_xlim(0, 1.01)
ax.set_ylim(-0.5, len(targets) - 0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [Patch(facecolor=color_map[f], edgecolor="black", label=f) for f in features_reduced]
ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=len(features_reduced),
    frameon=False,
    fontsize=12,
    title_fontsize=14,
)

plt.tight_layout(pad=2.0)
plt.savefig("feature_imp.png", dpi=600, bbox_inches="tight")
plt.show()

df_obj = df.iloc[:, -7:-1]

corr = df_obj.corr(method="spearman")

# Mask only the upper triangle (not the diagonal)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

plt.figure(figsize=(12, 12), dpi=300)
sns.set(
    style="white",
    font_scale=1.5,
    rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "axes.grid": False,
        "font.family": "sans-serif",
    },
)

ax = sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=2,
    cbar_kws={"shrink": 0.75, "aspect": 30, "pad": 0.03, "label": "Correlation"},
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14, weight="bold")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("correlation.png", dpi=600, bbox_inches="tight", transparent=False)

plt.show()
