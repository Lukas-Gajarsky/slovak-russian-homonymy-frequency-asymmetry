# -*- coding: utf-8 -*-
"""
Reproducible analysis for:

Frequency Asymmetry and Lexical Interference in
Slovak–Russian Interlingual Homonymy:
A Corpus-Based and Behavioral Study

Input files expected in the same directory as this script:
    01_frequency_dataset.xlsx
    02_behavioral_dataset.xlsx

The script reproduces the main descriptive and inferential analyses
and generates Figures 1–3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================
# 1. FILES AND OUTPUT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FREQUENCY_FILE = BASE_DIR / "01_frequency_dataset.xlsx"
BEHAVIORAL_FILE = BASE_DIR / "02_behavioral_dataset.xlsx"

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. FREQUENCY ASYMMETRY
# ============================================================

print("=" * 60)
print("FREQUENCY ASYMMETRY")
print("=" * 60)

freq = pd.read_excel(
    FREQUENCY_FILE,
    sheet_name="DATA_579"
)

required_frequency_columns = ["RU_IPM", "SK_IPM"]

missing = [
    col for col in required_frequency_columns
    if col not in freq.columns
]

if missing:
    raise ValueError(
        f"Missing frequency columns: {missing}"
    )


# Calculate log-ratio independently from the stored values.
epsilon = 0.1

freq["log_ratio_calc"] = np.log2(
    (freq["SK_IPM"] + epsilon) /
    (freq["RU_IPM"] + epsilon)
)

# If the source dataset contains an existing log_ratio column,
# compare it with the independently calculated value.
if "log_ratio" in freq.columns:
    max_difference = (
        freq["log_ratio_calc"] - freq["log_ratio"]
    ).abs().max()

    print(
        f"Maximum difference from stored log_ratio: "
        f"{max_difference:.12f}"
    )

freq["log_ratio"] = freq["log_ratio_calc"]


# Basic statistics
n_frequency = len(freq)
median = freq["log_ratio"].median()
minimum = freq["log_ratio"].min()
maximum = freq["log_ratio"].max()

print(f"N = {n_frequency}")
print(f"Minimum = {minimum:.3f}")
print(f"Median = {median:.3f}")
print(f"Maximum = {maximum:.3f}")


# Frequency-dominance classification
freq["dominance"] = np.select(
    [
        freq["log_ratio"] > 0.5,
        freq["log_ratio"] < -0.5
    ],
    [
        "SK-dominant",
        "RU-dominant"
    ],
    default="Balanced"
)

frequency_counts = freq["dominance"].value_counts()
frequency_percentages = (
    freq["dominance"]
    .value_counts(normalize=True)
    .mul(100)
)

frequency_summary = pd.DataFrame({
    "n": frequency_counts,
    "%": frequency_percentages.round(2)
})

frequency_summary = frequency_summary.reindex(
    ["RU-dominant", "SK-dominant", "Balanced"]
)

print("\nFrequency dominance:")
print(frequency_summary)


# ============================================================
# 3. FIGURE 1
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    freq["log_ratio"],
    bins=35,
    edgecolor="white",
    linewidth=0.7
)

ax.axvline(
    -0.5,
    linestyle="--",
    linewidth=1.2,
    label="Classification threshold = −0.5"
)

ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    label="Classification threshold = +0.5"
)

ax.axvline(
    median,
    linestyle=":",
    linewidth=1.5,
    label=f"Median = {median:.2f}"
)

ax.set_title(
    "Distribution of Frequency Asymmetry Across 579 Lexical Pairs",
    fontsize=14,
    pad=12
)

ax.set_xlabel(
    "Frequency asymmetry (log₂ SK/RU ratio)",
    fontsize=11
)

ax.set_ylabel(
    "Number of lexical pairs",
    fontsize=11
)

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5
)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="upper right"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_1_frequency_asymmetry.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_1_frequency_asymmetry.pdf",
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure 1 saved.")


# ============================================================
# 4. BEHAVIORAL DATA
# ============================================================

print("\n" + "=" * 60)
print("BEHAVIORAL DATA")
print("=" * 60)

behavior = pd.read_excel(
    BEHAVIORAL_FILE,
    sheet_name="MASTER_RESPONSES"
)

required_behavior_columns = [
    "participant_id",
    "item_id",
    "task",
    "error_type",
    "pair_id",
    "category",
    "valid_for_primary_model"
]

missing = [
    col for col in required_behavior_columns
    if col not in behavior.columns
]

if missing:
    raise ValueError(
        f"Missing behavioral columns: {missing}"
    )


# Keep responses included in the primary analysis.
valid = behavior[
    behavior["valid_for_primary_model"] == True
].copy()

print(f"Total recorded responses: {len(behavior)}")
print(f"Valid responses: {len(valid)}")
print(f"Excluded responses: {len(behavior) - len(valid)}")
print(f"Participants: {valid['participant_id'].nunique()}")
print(f"Valid items: {valid['item_id'].nunique()}")


# ============================================================
# 5. OVERALL RESPONSE CATEGORIES
# ============================================================

order = [
    "correct",
    "interference",
    "other",
    "blank"
]

counts = valid["error_type"].value_counts()
counts = counts.reindex(order, fill_value=0)

percentages = counts / len(valid) * 100

response_summary = pd.DataFrame({
    "n": counts,
    "%": percentages.round(2)
})

print("\nOverall response categories:")
print(response_summary)


# ============================================================
# 6. FIGURE 2
# ============================================================

labels = [
    "Correct",
    "Interference",
    "Other",
    "Blank"
]

values = [
    percentages["correct"],
    percentages["interference"],
    percentages["other"],
    percentages["blank"]
]

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(labels, values)

ax.set_ylabel("Percentage of responses")
ax.set_title(
    "Distribution of Response Categories Across Valid Responses",
    fontsize=14,
    pad=12
)

ax.set_ylim(0, max(values) * 1.20)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_2_response_categories.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_2_response_categories.pdf",
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure 2 saved.")


# ============================================================
# 7. INTERFERENCE BY TASK
# ============================================================

task_summary = (
    valid.groupby("task")["error_type"]
    .apply(lambda x: (x == "interference").sum())
    .to_frame("interference_n")
)

task_totals = (
    valid.groupby("task")
    .size()
    .to_frame("total_n")
)

task_summary = task_summary.join(task_totals)

task_summary["interference_%"] = (
    task_summary["interference_n"] /
    task_summary["total_n"] * 100
)

print("\nInterference by task:")
print(task_summary.round(2))


# ============================================================
# 8. PARTICIPANT-LEVEL INTERFERENCE RATES
# ============================================================

valid["interference_binary"] = (
    valid["error_type"] == "interference"
).astype(int)

participant_task = (
    valid
    .groupby(["participant_id", "task"])["interference_binary"]
    .mean()
    .unstack()
)

required_tasks = ["production", "reception"]

missing_tasks = [
    task for task in required_tasks
    if task not in participant_task.columns
]

if missing_tasks:
    raise ValueError(
        f"Missing task columns: {missing_tasks}"
    )

paired = participant_task[
    required_tasks
].dropna()

production = paired["production"]
reception = paired["reception"]


# ============================================================
# 9. WILCOXON SIGNED-RANK TEST
# ============================================================

result = wilcoxon(
    production,
    reception,
    alternative="two-sided",
    method="approx"
)

differences = production - reception

n_zero = int((differences == 0).sum())
n_nonzero = int((differences != 0).sum())

z = result.zstatistic
r = z / np.sqrt(n_nonzero)

print("\nWilcoxon signed-rank test:")
print(f"N participants = {len(paired)}")
print(f"W = {result.statistic}")
print(f"p = {result.pvalue:.6f}")
print(f"Zero differences = {n_zero}")
print(f"Non-zero paired differences = {n_nonzero}")
print(f"z = {z:.4f}")
print(f"r = {r:.4f}")
print(f"|r| = {abs(r):.4f}")


# Participant-level descriptive statistics
print("\nIndividual interference rates:")
print(
    f"Production: mean = {production.mean() * 100:.2f}%, "
    f"SD = {production.std() * 100:.2f}%"
)

print(
    f"Reception: mean = {reception.mean() * 100:.2f}%, "
    f"SD = {reception.std() * 100:.2f}%"
)


# ============================================================
# 10. FREQUENCY DOMINANCE AND INTERFERENCE
# ============================================================

dominant = valid[
    valid["category"].isin(
        ["SK-dominant", "RU-dominant"]
    )
].copy()

print("\nFrequency-dominance categories:")
print(
    valid.groupby("pair_id")["category"]
    .first()
    .value_counts()
)

print(
    f"\nNumber of valid responses in dominant groups: "
    f"{len(dominant)}"
)

print(
    f"Number of pairs: {dominant['pair_id'].nunique()}"
)


# Overall comparison
overall = (
    dominant
    .groupby("category")
    .agg(
        interference_n=(
            "error_type",
            lambda x: (x == "interference").sum()
        ),
        total_n=("error_type", "size")
    )
)

overall["interference_%"] = (
    overall["interference_n"] /
    overall["total_n"] * 100
)

print("\nOverall interference by frequency dominance:")
print(overall.round(2))


# By task and frequency dominance
by_task = (
    dominant
    .groupby(["task", "category"])
    .agg(
        interference_n=(
            "error_type",
            lambda x: (x == "interference").sum()
        ),
        total_n=("error_type", "size")
    )
)

by_task["interference_%"] = (
    by_task["interference_n"] /
    by_task["total_n"] * 100
)

print("\nInterference by task and frequency dominance:")
print(by_task.round(2))


# Publication-oriented table
result_table = by_task.reset_index()

result_table["n / N"] = (
    result_table["interference_n"].astype(str)
    + " / "
    + result_table["total_n"].astype(str)
)

result_table["Interference (%)"] = (
    result_table["interference_%"]
    .map(lambda x: f"{x:.2f}")
)

result_table = result_table[
    ["task", "category", "n / N", "Interference (%)"]
]

print("\nPublication table:")
print(result_table.to_string(index=False))


# ============================================================
# 11. CHECK EXCLUDED ITEMS
# ============================================================

excluded = behavior[
    behavior["valid_for_primary_model"] != True
].copy()

print("\nExcluded items:")

excluded_columns = [
    "item_id",
    "pair_id",
    "task",
    "category",
    "administered_stimulus"
]

available_excluded_columns = [
    col for col in excluded_columns
    if col in excluded.columns
]

print(
    excluded[available_excluded_columns]
    .drop_duplicates()
    .to_string(index=False)
)


# ============================================================
# 12. FIGURE 3
# ============================================================

task_order = [
    "production",
    "reception"
]

category_order = [
    "RU-dominant",
    "SK-dominant"
]

plot_data = by_task.reset_index()

plot_data["task"] = pd.Categorical(
    plot_data["task"],
    categories=task_order,
    ordered=True
)

plot_data["category"] = pd.Categorical(
    plot_data["category"],
    categories=category_order,
    ordered=True
)

plot_data = plot_data.sort_values(
    ["task", "category"]
)

production_values = (
    plot_data[plot_data["task"] == "production"]
    .set_index("category")
    .reindex(category_order)["interference_%"]
)

reception_values = (
    plot_data[plot_data["task"] == "reception"]
    .set_index("category")
    .reindex(category_order)["interference_%"]
)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

x = [0, 1]
width = 0.36

bars1 = ax.bar(
    [i - width / 2 for i in x],
    production_values,
    width,
    label="Slovak→Russian production"
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    reception_values,
    width,
    label="Russian→Slovak reception"
)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_title(
    "Lexical Interference by Frequency Dominance and Task",
    fontsize=14,
    pad=12
)

ax.set_ylabel("Interference responses (%)")
ax.set_xlabel("Frequency dominance")

ax.set_xticks(x)
ax.set_xticklabels([
    "RU-dominant",
    "SK-dominant"
])

ax.set_ylim(0, 55)

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5
)

ax.legend(
    frameon=False,
    fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_3_frequency_dominance_interference.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_3_frequency_dominance_interference.pdf",
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure 3 saved.")


# ============================================================
# 13. SAVE ANALYTICAL SUMMARIES
# ============================================================

frequency_summary.to_csv(
    OUTPUT_DIR / "frequency_dominance_summary.csv"
)

response_summary.to_csv(
    OUTPUT_DIR / "response_category_summary.csv"
)

task_summary.to_csv(
    OUTPUT_DIR / "interference_by_task.csv"
)

overall.to_csv(
    OUTPUT_DIR / "interference_by_frequency_dominance.csv"
)

result_table.to_csv(
    OUTPUT_DIR / "interference_by_task_and_frequency_dominance.csv",
    index=False
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print("\nFrequency dataset:")
print(f"  N = {n_frequency}")
print(f"  Median log-ratio = {median:.3f}")

print("\nBehavioral dataset:")
print(f"  Total responses = {len(behavior)}")
print(f"  Valid responses = {len(valid)}")
print(f"  Participants = {valid['participant_id'].nunique()}")
print(f"  Valid items = {valid['item_id'].nunique()}")

print("\nInterference:")
print(
    f"  Production = "
    f"{task_summary.loc['production', 'interference_%']:.2f}%"
)

print(
    f"  Reception = "
    f"{task_summary.loc['reception', 'interference_%']:.2f}%"
)

print("\nWilcoxon:")
print(f"  W = {result.statistic}")
print(f"  p = {result.pvalue:.6f}")
print(f"  |r| = {abs(r):.3f}")

print("\nOutput directory:")
print(OUTPUT_DIR)
