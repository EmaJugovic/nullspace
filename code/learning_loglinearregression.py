#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learning Curve Visualization with Logarithmic Regression Model: y = a + b*log(t)
- Each participant's data normalized individually (min=0, max=1) across 5 days
- Model: y = a + b*log(t) where t is day number (linear regression on log-transformed session values)
- Regression performed on normalized data for each participant
- Individual regression curves: coloured smooth lines
- Population statistics: median/mean across participants' regression fits
- X-axis: Days 1-5 (averaged across 5 blocks per day)
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, integrate
import re

DEFAULT_PROJECT_ROOT = Path(
    "/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis"
)
PROJECT_ROOT = Path(os.environ.get("NULLSPACE_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT)))
BEHAVIOUR_DIR = PROJECT_ROOT / "NullSpace_Data" / "Behaviour"

# ========== TASK CONFIGURATION ==========
TASKS = {
    "Tapes_Seated": {
        "paths": [
            str(BEHAVIOUR_DIR / "tapes_summary.csv"),
        ],
        "color": "#F0E442",  # yellow (Tapes seated)
        "value_col": "Transferred",
    },
    "Tapes_Standing": {
        "paths": [
            str(BEHAVIOUR_DIR / "tapes_standing_summary.csv"),
        ],
        "color": "#009E73",  # green (Tapes standing)
        "value_col": "Transferred",
    },
    "Coin_Task_Seated": {
        "paths": [
            str(BEHAVIOUR_DIR / "coin_task_summary.csv"),
        ],
        "color": "#7F7F7F",  # gray (Coins)
        "value_col": "Transferred",
    },
    "Pegs_Seated": {
        "paths": [
            str(BEHAVIOUR_DIR / "pegs_summary.csv"),
        ],
        "color": "#0072B2",  # blue (Pegs seated)
        "value_col": "Transferred",
    },
    "Pegs_Standing": {
        "paths": [
            str(BEHAVIOUR_DIR / "pegs_standing_summary.csv"),
        ],
        "color": "#fb8072",  # salmon/pink
        "value_col": "Transferred",
    },
    "Cognitive": {
        "paths": [
            str(BEHAVIOUR_DIR / "tapes_cog_load_summary.csv"),
        ],
        "color": "#E69F00",
        "value_col": "Transferred",
    },
}

# ========== SELECT WHICH TASK(S) TO VISUALIZE ==========
PLANNED_TASK_COMPARISONS = [
    ("tapes_vs_coins", "Tapes_Seated", "Coin_Task_Seated"),
    ("tapes_vs_tapes_standing", "Tapes_Seated", "Tapes_Standing"),
    ("tapes_vs_cognitive", "Tapes_Seated", "Cognitive"),
    ("tapes_vs_pegs", "Tapes_Seated", "Pegs_Seated"),
]
SELECTED_TASKS = list(dict.fromkeys(
    task for _, comparison_task, reference_task in PLANNED_TASK_COMPARISONS
    for task in (comparison_task, reference_task)
))

# ========== VISUALIZATION STYLE OPTIONS ==========
"""
Choose visualization style(s) for population data:
  "line_shaded"  : Line plot + shaded CI/SD band (original style)
  "bar_errorbar" : Bar plot + error bars for CI/SD (new style)
"""
VISUALIZATION_STYLES = ["line_shaded"]

# ========== STATISTICAL METHOD OPTIONS ==========
"""
Choose statistical method(s) for computing central tendency and error:
  "median_se"  : Median ± SE of median (SE_median = sqrt(π / (2*n)) * std)
  "mean_sd"    : Mean ± Standard Deviation (classical method)
  "mean_sem"   : Mean ± Standard Error of Mean (SEM = SD/sqrt(n))
  "mean_ci"    : Mean + 95% CI (t-distribution method)
"""
STATISTICAL_METHODS = ["median_se", "mean_sem"]

# ========== LABEL OPTIONS ==========
"""
Choose whether to label individual participants:
  True  : Add participant labels at the end of each curve, show only population stats in legend
  False : Show all participants in legend (traditional style)
"""
LABEL_INDIVIDUALS = False

EXCLUDED_PARTICIPANTS = {"PT02", "PT18", "PT19"}

OUT_DIR = Path(os.environ.get(
    "NULLSPACE_LEARNING_RESULTS_DIR",
    str(PROJECT_ROOT / "Results_2" / "C_learning_loglinear_pairwise_BF"),
))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DPI = int(os.environ.get("NULLSPACE_SAVE_DPI", "300"))

# --------- COLORS ----------
CLR_MEDIAN = "black"       # median line/bar
CLR_SEATED_BAR = "#87ceeb"  # sky blue for seated bar
CLR_STANDING_BAR = "#ffb6c1"  # light pink for standing bar
# Extended color palette for 26+ participants (distinct colors)
CLR_POINTS_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#9467bd", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d"
]  # 30 distinct colors for individual participants

# --------- HELPERS ----------
def load_first_available(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"None of the paths exist: {paths}")

def normalize_participant_id(participant_id):
    """Normalize participant ID for robust matching (e.g., PT_02 -> PT02)."""
    participant = re.sub(r"[^A-Za-z0-9]", "", str(participant_id)).upper()
    if participant.startswith("PT") and participant[2:].isdigit():
        return f"PT{int(participant[2:]):02d}"
    return participant

def bayes_factor_ttest(t_value, n_pairs, r_scale=np.sqrt(2) / 2):
    """Two-sided JZS Bayes factor BF10 for a paired t-test (Rouder et al.)."""
    if n_pairs < 2 or not np.isfinite(t_value):
        return np.nan
    degrees_freedom = n_pairs - 1

    def integrand(g):
        prior = (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g))
        scaled = 1 + n_pairs * g * r_scale ** 2
        likelihood = scaled ** -0.5
        likelihood *= (
            1 + t_value ** 2 / (degrees_freedom * scaled)
        ) ** (-(degrees_freedom + 1) / 2)
        return likelihood * prior

    numerator, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    denominator = (
        1 + t_value ** 2 / degrees_freedom
    ) ** (-(degrees_freedom + 1) / 2)
    return numerator / denominator

def interpret_bf10(bf10):
    """Return a compact, symmetric interpretation of BF10."""
    if not np.isfinite(bf10):
        return "not estimable"
    if bf10 >= 100:
        return "extreme evidence for H1"
    if bf10 >= 30:
        return "very strong evidence for H1"
    if bf10 >= 10:
        return "strong evidence for H1"
    if bf10 >= 3:
        return "moderate evidence for H1"
    if bf10 >= 1:
        return "anecdotal evidence for H1"
    if bf10 > 1 / 3:
        return "anecdotal evidence for H0"
    if bf10 > 1 / 10:
        return "moderate evidence for H0"
    return "strong evidence for H0"

def add_x_index(df):
    """Add running x index: Day1 .. Day5 => x = 1..5."""
    out = df.copy()
    out["Day"] = out["Day"].astype(int)
    out["Block"] = out["Block"].astype(int)
    out["x"] = out["Day"]
    return out

def normalize_participant_data(df, value_col="Transferred"):
    """
    Apply Min-Max normalization to each participant at day level:
    1. First compute daily averages (mean across 5 blocks per day)
    2. Then normalize these 5 daily means: (value - min) / (max - min)
    Where min and max are computed per participant across the 5 days.
    """
    # First, compute daily averages for each participant
    daily_avg = df.groupby(["Participant", "Day"])[value_col].mean().reset_index()
    daily_avg.rename(columns={value_col: f"{value_col}_daily_avg"}, inplace=True)
    
    # Transform keeps normalized values aligned with their original rows.
    daily_value_col = f"{value_col}_daily_avg"
    participant_min = daily_avg.groupby("Participant")[daily_value_col].transform("min")
    participant_max = daily_avg.groupby("Participant")[daily_value_col].transform("max")
    participant_range = participant_max - participant_min
    daily_avg[f"{value_col}_normalized"] = np.where(
        participant_range == 0,
        0.5,
        (daily_avg[daily_value_col] - participant_min) / participant_range,
    )
    # Add x column for plotting (x = Day)
    daily_avg["x"] = daily_avg["Day"]
    return daily_avg

def log_linear_regression(x, y, x_pred=None):
    """
    Perform logarithmic regression: y = a + b*log(t)
    Linear regression on log-transformed session (time) values.
    Args:
        x: training x values (time/days)
        y: training y values (performance)
        x_pred: x values for prediction (if None, use x)
    Returns: intercept (a), slope (b), predicted values at x_pred positions
    """
    # Log transform x values (time/session)
    x_log = np.log(x)
    
    # Perform linear regression on (log(x), y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y)
    
    # Use x_pred if provided, otherwise use x
    if x_pred is None:
        x_pred = x
    
    # Predict: y = a + b*log(t)
    x_pred_log = np.log(x_pred)
    y_pred = intercept + slope * x_pred_log
    
    return intercept, slope, y_pred

def process_participant_data(df, value_col="Transferred", n_smooth_points=100):
    """
    Process participant data:
    1. Compute daily averages and normalize per participant (mean across 5 blocks per day)
    2. Perform logarithmic regression y=a+b*log(t) on NORMALIZED data for each participant
    3. Return both daily stats (5 points) and smooth curves (n_smooth_points)
    """
    # First, normalize the data per participant (includes daily averaging)
    daily_avg = normalize_participant_data(df, value_col)
    normalized_col = f"{value_col}_normalized"
    
    # For statistics: regression at original 5 day points
    regression_values = []
    # For smooth curves: store separately
    smooth_curves = {}
    model_parameters = []
    
    # Create dense x values for smooth plotting (e.g., 100 points from Day 1 to Day 5)
    x_smooth = np.linspace(1, 5, n_smooth_points)
    
    for participant in daily_avg["Participant"].unique():
        mask = daily_avg["Participant"] == participant
        participant_data = daily_avg.loc[mask].sort_values("Day")
        
        x = participant_data["Day"].values
        y = participant_data[normalized_col].values  # Use normalized values
        
        # Perform logarithmic regression at original points (for stats)
        intercept, slope, y_pred = log_linear_regression(x, y, x_pred=x)
        regression_values.extend(y_pred)
        residual_sum_squares = float(np.sum((y - y_pred) ** 2))
        total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = (
            1 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0 else np.nan
        )
        model_parameters.append({
            "Participant": participant,
            "intercept": intercept,
            "slope": slope,
            "r_squared": r_squared,
            "n_days": len(x),
            "days_used": ",".join(str(int(day)) for day in x),
        })
        
        # Generate smooth curve (for plotting)
        _, _, y_smooth = log_linear_regression(x, y, x_pred=x_smooth)
        smooth_curves[participant] = {
            "x": x_smooth.copy(),
            "y": y_smooth.copy()
        }
    
    daily_avg[f"{value_col}_regression"] = regression_values
    daily_avg["x"] = daily_avg["Day"]
    
    return daily_avg, smooth_curves, pd.DataFrame(model_parameters)

def compute_se_median(values):
    """Compute Standard Error of the median.
    SE_median = sqrt(π / (2 * n)) * std
    Return NaN when data are missing/insufficient.
    """
    values = np.array(values)
    n = len(values)
    if n < 2:
        return np.nan
    
    std = np.std(values, ddof=1)
    se_median = np.sqrt(np.pi / (2 * n)) * std
    return se_median

def compute_mean_sd(values):
    """Compute mean and standard deviation."""
    values = np.array(values)
    n = len(values)
    if n < 1:
        return np.nan, np.nan
    
    mean = np.mean(values)
    if n < 2:
        sd = 0.0
    else:
        sd = np.std(values, ddof=1)  # sample SD
    
    return mean - sd, mean + sd

def compute_mean_sem(values):
    """Compute mean and Standard Error of Mean (SEM = SD/sqrt(n))."""
    values = np.array(values)
    n = len(values)
    if n < 1:
        return np.nan, np.nan
    
    mean = np.mean(values)
    if n < 2:
        sem = 0.0
    else:
        sd = np.std(values, ddof=1)
        sem = sd / np.sqrt(n)
    
    return mean - sem, mean + sem

def compute_mean_ci(values):
    """Compute mean and 95% CI using t-distribution."""
    values = np.array(values)
    n = len(values)
    if n < 1:
        return np.nan, np.nan
    
    mean = np.mean(values)
    if n < 2:
        return mean, mean  # no CI if n<2
    
    sd = np.std(values, ddof=1)
    sem = sd / np.sqrt(n)
    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, n - 1)
    margin = t_crit * sem
    
    return mean - margin, mean + margin

def day_stats(df, value_col="Transferred_regression", method="median_se"):
    """Compute statistics per Day using specified method on regression values."""
    stats_list = []
    for day, group_df in df.groupby("Day"):
        values = group_df[value_col].dropna().values
        x = int(day)
        if len(values) == 0:
            stats_list.append({
                "Day": int(day),
                "x": x,
                "central": np.nan,
                "lower": np.nan,
                "upper": np.nan,
            })
            continue
        
        if method == "median_se":
            median = np.median(values)
            se_median = compute_se_median(values)
            central = median
            ci_lower = median - se_median
            ci_upper = median + se_median
        elif method == "mean_sd":
            central = np.mean(values)
            if len(values) < 2:
                ci_lower = central
                ci_upper = central
            else:
                sd = np.std(values, ddof=1)
                ci_lower = central - sd
                ci_upper = central + sd
        elif method == "mean_sem":
            central = np.mean(values)
            ci_lower, ci_upper = compute_mean_sem(values)
        elif method == "mean_ci":
            central = np.mean(values)
            ci_lower, ci_upper = compute_mean_ci(values)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        stats_list.append({
            "Day": int(day),
            "x": x,
            "central": central,
            "lower": ci_lower,
            "upper": ci_upper,
        })
    
    return pd.DataFrame(stats_list).sort_values("Day").reset_index(drop=True)

def get_individual_trajectories(df, value_col="Transferred_regression"):
    """Get x, y per participant at day level (regression curves)."""
    trajectories = {}
    for participant, group_df in df.groupby("Participant"):
        group_df = group_df.sort_values("Day").reset_index(drop=True)
        trajectories[participant] = {
            "x": group_df["x"].values,
            "y": group_df[value_col].values,
        }
    return trajectories

# ========== LOAD AND PROCESS DATA ==========
all_data = {}

for task_name in SELECTED_TASKS:
    if task_name not in TASKS:
        print(f"Warning: Task '{task_name}' not found in TASKS dict")
        continue
    
    task_info = TASKS[task_name]
    value_col = task_info.get("value_col", "Transferred")  # default to Transferred
    
    # Load data
    p_task = load_first_available(task_info["paths"])
    df_task = pd.read_csv(p_task)
    df_task["Participant"] = df_task["Participant"].map(normalize_participant_id)

    # Exclude selected participants (case-insensitive, robust to separators)
    excluded_norm = {normalize_participant_id(p) for p in EXCLUDED_PARTICIPANTS}
    participant_norm = df_task["Participant"].map(normalize_participant_id)
    keep_mask = ~participant_norm.isin(excluded_norm)
    removed_participants = sorted(df_task.loc[~keep_mask, "Participant"].astype(str).unique().tolist())
    df_task = df_task.loc[keep_mask].copy()
    if removed_participants:
        print(f"{task_name}: Excluded participants -> {removed_participants}")
    
    # Basic column check
    required_cols = {"Participant", "Day", "Block", value_col}
    missing = required_cols - set(df_task.columns)
    if missing:
        raise ValueError(f"Task '{task_name}' file missing columns: {missing}")
    
    # Clean + add x index
    df_task = add_x_index(df_task)
    
    # Process data: compute daily averages and perform log-linear regression
    # Returns both daily stats and smooth curves for plotting
    df_task, smooth_curves, model_parameters = process_participant_data(
        df_task, value_col, n_smooth_points=100
    )
    regression_col = f"{value_col}_regression"
    
    # Compute stats for each statistical method
    stats_by_method = {}
    for stat_method in STATISTICAL_METHODS:
        stats_by_method[stat_method] = day_stats(df_task, regression_col, method=stat_method)
    
    trajectories = get_individual_trajectories(df_task, regression_col)
    
    all_data[task_name] = {
        "df": df_task,
        "stats_by_method": stats_by_method,
        "trajectories": trajectories,
        "smooth_curves": smooth_curves,  # Add smooth curves for plotting
        "model_parameters": model_parameters,
        "color": task_info["color"],
        "value_col": regression_col,
    }
    
    # Debug: show participant IDs
    participant_list = sorted(trajectories.keys())
    print(f"Loaded {task_name}: {len(trajectories)} participants (using regression column: {regression_col})")
    print(f"  Participants: {participant_list}")

if not all_data:
    raise ValueError("No tasks loaded successfully!")

# ========== SAVE MODEL DATA AND RUN PRESPECIFIED PAIRWISE SLOPE COMPARISONS ==========
model_parameter_table = pd.concat(
    [
        task_data["model_parameters"].assign(Task=task_name)
        for task_name, task_data in all_data.items()
    ],
    ignore_index=True,
)
model_parameter_table.to_csv(
    OUT_DIR / "participant_loglinear_parameters.csv", index=False
)

daily_model_table = pd.concat(
    [
        task_data["df"].assign(Task=task_name)
        for task_name, task_data in all_data.items()
    ],
    ignore_index=True,
)
daily_model_table.to_csv(OUT_DIR / "participant_daily_model_values.csv", index=False)

pairwise_rows = []
paired_value_rows = []
for comparison_name, task_a, task_b in PLANNED_TASK_COMPARISONS:
    slopes_a = (
        all_data[task_a]["model_parameters"]
        .set_index("Participant")["slope"]
        .sort_index()
    )
    slopes_b = (
        all_data[task_b]["model_parameters"]
        .set_index("Participant")["slope"]
        .sort_index()
    )
    common_participants = slopes_a.index.intersection(slopes_b.index).sort_values()
    if len(common_participants) < 2:
        raise ValueError(
            f"{comparison_name}: fewer than two paired participants "
            f"(n={len(common_participants)})"
        )

    values_a = slopes_a.loc[common_participants].to_numpy(dtype=float)
    values_b = slopes_b.loc[common_participants].to_numpy(dtype=float)
    differences = values_a - values_b
    n_pairs = len(differences)
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    if sd_difference == 0 and mean_difference == 0:
        t_statistic, p_value = 0.0, 1.0
    else:
        t_statistic, p_value = stats.ttest_rel(values_a, values_b)
        t_statistic, p_value = float(t_statistic), float(p_value)

    sem_difference = sd_difference / np.sqrt(n_pairs)
    t_critical = stats.t.ppf(0.975, n_pairs - 1)
    ci_low = mean_difference - t_critical * sem_difference
    ci_high = mean_difference + t_critical * sem_difference
    cohen_dz = mean_difference / sd_difference if sd_difference > 0 else np.nan
    bf10 = bayes_factor_ttest(t_statistic, n_pairs)
    bf01 = 1 / bf10 if np.isfinite(bf10) and bf10 > 0 else np.nan
    shapiro_p = (
        float(stats.shapiro(differences).pvalue)
        if 3 <= n_pairs <= 5000 and np.ptp(differences) > 0
        else np.nan
    )
    if np.allclose(differences, 0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            differences, alternative="two-sided", method="auto"
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)

    pairwise_rows.append({
        "comparison": comparison_name,
        "task_a": task_a,
        "task_b": task_b,
        "difference_direction": f"{task_a} - {task_b}",
        "n_pairs": n_pairs,
        "mean_slope_task_a": float(np.mean(values_a)),
        "mean_slope_task_b": float(np.mean(values_b)),
        "mean_difference": mean_difference,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": cohen_dz,
        "t_statistic": t_statistic,
        "degrees_freedom": n_pairs - 1,
        "p_raw_two_sided": p_value,
        "BF10_JZS_two_sided": bf10,
        "BF01_JZS_two_sided": bf01,
        "BF10_interpretation": interpret_bf10(bf10),
        "shapiro_p_difference": shapiro_p,
        "normality_flag_shapiro_p_lt_0_05": bool(
            np.isfinite(shapiro_p) and shapiro_p < 0.05
        ),
        "wilcoxon_statistic_sensitivity": wilcoxon_statistic,
        "wilcoxon_p_raw_sensitivity": wilcoxon_p,
    })
    for participant, value_a, value_b, difference in zip(
        common_participants, values_a, values_b, differences
    ):
        paired_value_rows.append({
            "comparison": comparison_name,
            "Participant": participant,
            "task_a": task_a,
            "task_b": task_b,
            "slope_task_a": value_a,
            "slope_task_b": value_b,
            "difference_task_a_minus_task_b": difference,
        })

pairwise_results = pd.DataFrame(pairwise_rows)
pairwise_results.to_csv(
    OUT_DIR / "planned_pairwise_learning_rate_bayes_results.csv", index=False
)
paired_values = pd.DataFrame(paired_value_rows)
paired_values.to_csv(
    OUT_DIR / "planned_pairwise_participant_slopes.csv", index=False
)

methods_text = """Analysis unit: participant.
Outcome: participant-specific slope b from y = a + b*log(day), fitted to daily means after within-participant min-max normalization.
Planned comparisons: Tapes_Seated minus Coin_Task_Seated, Tapes_Standing, Cognitive, and Pegs_Seated.
Frequentist analysis: two-sided paired t-test on matched participant slopes; 95% t confidence interval for the mean paired difference; Cohen's dz. Each task comparison addresses a distinct prespecified scientific hypothesis, so its two-sided P value is reported without cross-comparison multiplicity correction.
Bayesian analysis: two-sided Jeffreys-Zellner-Siow paired t-test Bayes factor, BF10, with Cauchy scale r=sqrt(2)/2. BF01=1/BF10.
Assumption/sensitivity analysis: Shapiro-Wilk test of paired differences and an unadjusted two-sided Wilcoxon signed-rank test for each prespecified comparison. Where paired differences are non-normal, treat the t-test and JZS t-test Bayes factor cautiously and use the Wilcoxon result as a robustness check.
Exclusions: PT02, PT18, and PT19, applied before model fitting as prespecified in this script.
The cognitive task contributes observed days 1, 3, and 5 only. No missing days are imputed.
Bayes factors quantify evidence for a non-zero versus zero mean paired slope difference; they are not probabilities that either hypothesis is true.
"""
(OUT_DIR / "analysis_methods_and_guardrails.txt").write_text(methods_text, encoding="utf-8")

# Paired learning-rate plots, one per planned comparison plus a 2x2 summary.
bayes_plot_dir = OUT_DIR / "planned_pairwise_bayes"
bayes_plot_dir.mkdir(parents=True, exist_ok=True)
fig_summary, axes_summary = plt.subplots(2, 2, figsize=(13, 11))
for ax_summary, result_row in zip(axes_summary.flat, pairwise_rows):
    comparison_name = result_row["comparison"]
    task_a = result_row["task_a"]
    task_b = result_row["task_b"]
    plot_df = paired_values.loc[paired_values["comparison"] == comparison_name]
    result = pairwise_results.loc[
        pairwise_results["comparison"] == comparison_name
    ].iloc[0]

    def draw_paired_slope_plot(ax):
        for _, participant_row in plot_df.iterrows():
            ax.plot(
                [0, 1],
                [participant_row["slope_task_a"], participant_row["slope_task_b"]],
                color="#9a9a9a", alpha=0.45, linewidth=1.0, zorder=1,
            )
        ax.scatter(
            np.zeros(len(plot_df)), plot_df["slope_task_a"],
            color=TASKS[task_a]["color"], s=34, alpha=0.8, zorder=2,
        )
        ax.scatter(
            np.ones(len(plot_df)), plot_df["slope_task_b"],
            color=TASKS[task_b]["color"], s=34, alpha=0.8, zorder=2,
        )
        for x_position, column, color in (
            (0, "slope_task_a", TASKS[task_a]["color"]),
            (1, "slope_task_b", TASKS[task_b]["color"]),
        ):
            values = plot_df[column].to_numpy(dtype=float)
            sem = stats.sem(values)
            margin = stats.t.ppf(0.975, len(values) - 1) * sem
            ax.errorbar(
                x_position, np.mean(values), yerr=margin, fmt="D",
                color="black", markerfacecolor=color, markeredgecolor="black",
                capsize=5, markersize=8, linewidth=2.0, zorder=4,
            )
        p_value = result["p_raw_two_sided"]
        p_label = "<0.001" if p_value < 0.001 else f"={p_value:.3f}"
        bf10 = result["BF10_JZS_two_sided"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            task_a.replace("_", " "), task_b.replace("_", " ")
        ], rotation=12, ha="right")
        ax.set_ylabel("Log-linear learning-rate slope (b)")
        ax.set_title(
            f"n={int(result['n_pairs'])}; paired t-test p{p_label}; BF10={bf10:.3g}",
            fontsize=10,
        )
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    draw_paired_slope_plot(ax_summary)
    fig_pair, ax_pair = plt.subplots(figsize=(7, 7))
    draw_paired_slope_plot(ax_pair)
    fig_pair.suptitle(comparison_name.replace("_", " "), fontweight="bold")
    fig_pair.tight_layout()
    fig_pair.savefig(
        bayes_plot_dir / f"{comparison_name}_paired_learning_rate_BF.png",
        dpi=SAVE_DPI, bbox_inches="tight",
    )
    plt.close(fig_pair)

fig_summary.suptitle(
    "Four planned paired learning-rate comparisons", fontsize=16, fontweight="bold"
)
fig_summary.tight_layout()
fig_summary.savefig(
    OUT_DIR / "planned_pairwise_learning_rate_BF_summary.png",
    dpi=SAVE_DPI, bbox_inches="tight",
)
plt.close(fig_summary)

print("\nPlanned pairwise learning-rate tests:")
print(pairwise_results.to_string(index=False))

# ========== PLOT ==========
for task_name, task_data in all_data.items():
    trajectories = task_data["trajectories"]
    smooth_curves = task_data["smooth_curves"]  # Use smooth curves for plotting
    stats_by_method = task_data["stats_by_method"]
    
    # Keep the population summary in the canonical task colour.
    population_color = task_data["color"]
    
    # Generate plot for each statistical method
    for stat_method in STATISTICAL_METHODS:
        stats_task = stats_by_method[stat_method]
        
        # Get data for this method
        xs = stats_task["x"].to_numpy()
        centrals = stats_task["central"].to_numpy()
        lowers = stats_task["lower"].to_numpy()
        uppers = stats_task["upper"].to_numpy()
        errors = np.array([centrals - lowers, uppers - centrals])
        
        # Method label for title and filename
        if stat_method == "median_se":
            method_label = "Median ± SE"
            method_abbrev = "median_se"
        elif stat_method == "mean_sd":
            method_label = "Mean ± SD"
            method_abbrev = "mean_sd"
        elif stat_method == "mean_sem":
            method_label = "Mean ± SEM"
            method_abbrev = "mean_sem"
        elif stat_method == "mean_ci":
            method_label = "Mean ± 95% CI"
            method_abbrev = "mean_ci"
        else:
            method_label = stat_method
            method_abbrev = stat_method
        
        # Generate plot for each visualization style
        for vis_style in VISUALIZATION_STYLES:
            fig, ax = plt.subplots(figsize=(12, 9))
            
            # Plot population data based on selected style (BEFORE individual participants)
            if vis_style == "line_shaded":
                # Style 1: Line + shaded error band
                ax.plot(xs, centrals, marker="D", linewidth=2.5, markersize=7,
                        color=population_color, label=method_label, zorder=5)
                ax.fill_between(xs, lowers, uppers, alpha=0.25, color=population_color)
                
            elif vis_style == "bar_errorbar":
                # Style 2: Bar + error bars
                ax.bar(xs, centrals, width=0.6, color=population_color, alpha=0.7, 
                       label=method_label, zorder=2)
                ax.errorbar(xs, centrals, yerr=errors, fmt="none", ecolor=population_color, 
                            capsize=5, capthick=2, linewidth=2, zorder=3)
            else:
                raise ValueError(f"Unknown visualization style: {vis_style}")
            
            # Plot individual participants with different colors (ON TOP)
            # Use smooth curves (100 points) for smooth lines
            for idx, (participant, curve) in enumerate(smooth_curves.items()):
                color = CLR_POINTS_PALETTE[idx % len(CLR_POINTS_PALETTE)]
                # Add label in legend only if not labeling on curve
                label = None if LABEL_INDIVIDUALS else f"{participant}"
                ax.plot(curve["x"], curve["y"], linewidth=2.0, 
                        alpha=0.6, color=color, label=label, zorder=4)
                
                # Add participant label at the end of the curve if enabled
                if LABEL_INDIVIDUALS:
                    x_end = curve["x"][-1]
                    y_end = curve["y"][-1]
                    ax.text(x_end + 0.05, y_end, f"{participant}", 
                            fontsize=8, color=color, va='center', 
                            fontweight='bold', alpha=0.8)
            
            # X-axis: label days
            xticks = [1, 2, 3, 4, 5]
            ax.set_xticks(xticks)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            ax.set_xlim(1, 5)
            # Set y-axis range to 0-1 (normalized data)
            ax.set_ylim(0, 1)
            
            ax.set_xlabel("Training Day (Days 1-5, averaged across 5 blocks per day)", fontsize=11)
            ax.set_ylabel("Normalized Performance (Model: y = a + b*log(t))", fontsize=11)
            ax.set_title(f"{task_name.replace('_', ' ')} - Learning Curve ({method_label})\n(Individual Min-Max Normalized; Coloured lines = logarithmic regression y=a+b*log(t); Task colour = population {method_label})", 
                         fontsize=12, fontweight="bold")
            
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(2.5)
            ax.spines['left'].set_linewidth(2.5)
            
            plt.tight_layout()
            
            # Save figure with method in filename
            out_fig = OUT_DIR / f"{task_name}_learning_curve_{method_abbrev}_{vis_style}.png"
            plt.savefig(out_fig, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"Saved: {out_fig.resolve()}")
            plt.close(fig)

# ========== COMBINED POSTER FIGURE: Augmentation (Tapes_Seated) + Tool Use (Coin_Task_Seated) ==========
COMBINED_TASK_PAIRS = [
    ("Tapes_Seated",    "Augmentation"),
    ("Coin_Task_Seated", "Tool Use"),
]

if all(task_name in all_data for task_name, _ in COMBINED_TASK_PAIRS):
    for stat_method in STATISTICAL_METHODS:
        # Resolve method label/abbreviation
        _method_labels = {
            "median_se": ("Median ± SE",    "median_se"),
            "mean_sd":   ("Mean ± SD",      "mean_sd"),
            "mean_sem":  ("Mean ± SEM",     "mean_sem"),
            "mean_ci":   ("Mean ± 95% CI",  "mean_ci"),
        }
        method_label, method_abbrev = _method_labels.get(stat_method, (stat_method, stat_method))

        for vis_style in VISUALIZATION_STYLES:
            fig, axes = plt.subplots(1, 2, figsize=(22, 9))

            for ax, (task_name, subplot_title) in zip(axes, COMBINED_TASK_PAIRS):
                task_data   = all_data[task_name]
                population_color = task_data["color"]
                smooth_curves_t = task_data["smooth_curves"]
                stats_task  = task_data["stats_by_method"][stat_method]

                xs       = stats_task["x"].to_numpy()
                centrals = stats_task["central"].to_numpy()
                lowers   = stats_task["lower"].to_numpy()
                uppers   = stats_task["upper"].to_numpy()
                errors   = np.array([centrals - lowers, uppers - centrals])

                # Population statistics
                if vis_style == "line_shaded":
                    ax.plot(xs, centrals, marker="D", linewidth=2.5, markersize=7,
                            color=population_color, label=method_label, zorder=5)
                    ax.fill_between(xs, lowers, uppers, alpha=0.25, color=population_color)
                elif vis_style == "bar_errorbar":
                    ax.bar(xs, centrals, width=0.6, color=population_color, alpha=0.7,
                           label=method_label, zorder=2)
                    ax.errorbar(xs, centrals, yerr=errors, fmt="none", ecolor=population_color,
                                capsize=5, capthick=2, linewidth=2, zorder=3)

                # Individual curves — NO legend labels
                for idx, (participant, curve) in enumerate(smooth_curves_t.items()):
                    color = CLR_POINTS_PALETTE[idx % len(CLR_POINTS_PALETTE)]
                    ax.plot(curve["x"], curve["y"], linewidth=2.0,
                            alpha=0.6, color=color, label=None, zorder=4)

                # X-axis ticks
                xticks = [1, 2, 3, 4, 5]
                ax.set_xticks(xticks)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                ax.set_xlim(1, 5)
                ax.set_ylim(0, 1)

                # No axis labels
                ax.set_xlabel("")
                ax.set_ylabel("")

                # Double the default tick label size (default ~10 → 20)
                ax.tick_params(axis="both", labelsize=20)

                ax.set_title(subplot_title, fontsize=18, fontweight="bold")
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_linewidth(2.5)
                ax.spines['left'].set_linewidth(2.5)

            plt.tight_layout()
            out_fig = OUT_DIR / f"combined_augmentation_tooluse_{method_abbrev}_{vis_style}.png"
            plt.savefig(out_fig, dpi=SAVE_DPI, bbox_inches="tight")
            print(f"Saved: {out_fig.resolve()}")
            plt.close(fig)
else:
    print("Warning: Tapes_Seated or Coin_Task_Seated not in all_data; skipping combined figure.")

# ========== FOUR PRESPECIFIED PAIRWISE LEARNING-CURVE FIGURES ==========
for comparison_name, comparison_task, reference_task in PLANNED_TASK_COMPARISONS:
    if comparison_task not in all_data or reference_task not in all_data:
        print(f"Warning: missing data for {comparison_name}; skipping pairwise figure.")
        continue
    for stat_method in STATISTICAL_METHODS:
        method_label, method_abbrev = {
            "median_se": ("Median +/- SE", "median_se"),
            "mean_sd": ("Mean +/- SD", "mean_sd"),
            "mean_sem": ("Mean +/- SEM", "mean_sem"),
            "mean_ci": ("Mean +/- 95% CI", "mean_ci"),
        }.get(stat_method, (stat_method, stat_method))
        for vis_style in VISUALIZATION_STYLES:
            fig, ax = plt.subplots(figsize=(9, 7))
            for task_name in (comparison_task, reference_task):
                task_data = all_data[task_name]
                stats_task = task_data["stats_by_method"][stat_method]
                xs = stats_task["x"].to_numpy()
                center = stats_task["central"].to_numpy()
                lower = stats_task["lower"].to_numpy()
                upper = stats_task["upper"].to_numpy()
                color = task_data["color"]
                label = task_name.replace("_", " ")
                if vis_style == "line_shaded":
                    ax.plot(xs, center, marker="D", linewidth=2.5, color=color, label=label)
                    ax.fill_between(xs, lower, upper, alpha=0.22, color=color)
                elif vis_style == "bar_errorbar":
                    offset = -0.14 if task_name == comparison_task else 0.14
                    ax.errorbar(
                        xs + offset, center,
                        yerr=np.vstack([center - lower, upper - center]),
                        fmt="o", linewidth=2, capsize=4, color=color, label=label,
                    )
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Training day")
            ax.set_ylabel("Normalized performance")
            ax.set_title(f"{comparison_name}: {method_label}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            pair_dir = OUT_DIR / "planned_pairwise" / comparison_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            out_fig = pair_dir / f"learning_curve_{comparison_name}_{method_abbrev}_{vis_style}.png"
            fig.savefig(out_fig, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_fig.resolve()}")

print("\n=== Learning Curve Generation Complete ===")
print(f"All figures saved to: {OUT_DIR.resolve()}")
