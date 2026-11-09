#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Smoothness Analysis
=========================================================
Multi-task, multi-phase, multi-joint analysis across 6 tasks.

- Segmentation: Four-stage method using WRIST_y thresholds
  * Grasp:      THRESHOLD_LOW↓ (from prev cycle) → valley → THRESHOLD_LOW↑
  * Carrying:   THRESHOLD_LOW↑ → THRESHOLD_HIGH↑
  * Placement:  THRESHOLD_HIGH↑ → peak
  * Return:     peak → THRESHOLD_LOW↓
- Metric: standard speed-profile LDLJ of the three-dimensional WRIST trajectory
- Visualization: Line+shaded and bar+errorbar styles
- Statistics: median_ci, mean_sd, mean_sem, mean_ci
- Output: D:\NullSpace_Analysis\Results_2\Smoothness_Analysis\
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from grasp_kinematic_common import (
	make_task_segments as _common_make_task_segments,
	task_name_candidates as _shared_task_name_candidates,
	trial_csv_path as _shared_trial_csv_path,
)


# =========================
# User parameters
# =========================
def resolve_group_path(path_like: str) -> Path:
	"""Use /group/plasticity on Linux and the equivalent UNC path on Windows."""
	prefix = "/group/plasticity/"
	if os.name == "nt" and path_like.startswith(prefix):
		rel = path_like[len(prefix):].replace("/", "\\")
		return Path(r"\\cbsu\data\Group\Plasticity") / rel
	return Path(path_like)


absref_root = resolve_group_path(
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/NullSpace_Data/Checked_Absref_data"
)

# ========== TASK CONFIGURATION ==========
TASKS = {
	"Tapes_Seated": {"task_name": "Tapes"},
	"Tapes_Standing": {"task_name": "Tapes_Standing"},
	"Pegs_Seated": {"task_name": "Pegs"},
	"Pegs_Standing": {"task_name": "Pegs_Standing"},
	"Coin_Task_Seated": {"task_name": "Coins"},

	"Cognitive": {"task_name": "Cognitive"},
}

# Subjects where seated task folders use *_First naming.
SUBJECTS_WITH_FIRST_NAMING = {"PT_23", "PT_24", "PT_25", "PT_26"}

# ========== SELECT WHICH TASK(S) TO ANALYZE ==========
# SELECTED_TASKS = ["Tapes_Seated", "Tapes_Standing", "Cognitive", "Coin_Task_Seated", "Pegs_Seated", "Pegs_Standing"]
SELECTED_TASKS = ["Tapes_Seated", "Tapes_Standing", "Pegs_Seated", "Pegs_Standing", "Coin_Task_Seated", "Cognitive"]

# Markers to analyze (3D positions)
MARKERS = ["WRIST"]

subject_list = ["PT_06", "PT_07", "PT_08", "PT_09", "PT_10", "PT_11", "PT_12", "PT_13", "PT_14",
				"PT_15", "PT_16", "PT_17", "PT_18", "PT_19", "PT_20", "PT_21", "PT_22", "PT_23",
				"PT_24", "PT_25", "PT_26"]
day_list = [1, 2, 3, 4, 5]
TASK_AVAILABLE_DAYS = {"Cognitive": [1, 3, 5]}
block_list = [1, 2, 3, 4, 5]

# Visualization settings
VISUALIZATION_STYLES = ["sparc_style"]  # SPARC-style group day line plots only; bar plots disabled

# ========== STATISTICAL METHOD OPTIONS ==========
"""
Choose statistical methods for computing central tendency and error:
  "median_se"  : Median +/- SE of median (SE_median = 1.253 * SD / sqrt(n))
  "mean_sd"    : Mean ± Standard Deviation (classical method)
  "mean_sem"   : Mean ± Standard Error of Mean (SEM = SD/sqrt(n))
  "mean_ci"    : Mean + 95% CI (t-distribution method)
"""
STATISTICAL_METHODS = ["median_se", "mean_sem"]
# Day-level within-participant aggregation across blocks in the same day
# "median": per subject-day median across available blocks
# "mean"  : per subject-day mean across available blocks
DAY_LEVEL_WITHIN_METHODS = ["median", "mean"]
SEGMENT_PHASES = ["grasp", "right_to_left_grasp"]

# Output directory
OUT_DIR = resolve_group_path("/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/Results_2") / "Smoothness_LDJK_Pairwise_Task_Comparisons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Segmentation settings
fs = 60.0
dt = 1.0 / fs
use_lowpass = True
lowpass_cutoff_hz = 8.0
lowpass_order = 2
marker_seg = "WRIST_y"  # segmentation marker
prominence_factor = 0.6
min_distance_s = 0.3
min_seg_len_frames = 10

# Four-stage segmentation thresholds
THRESHOLD_LOW = -50
THRESHOLD_HIGH = 180

# Jerk settings
jerk_pos_lowpass_hz = 8.0  # Optional low-pass on position before differentiation
LDLJ_IMPLEMENTATION_VERSION = "speed_3d_peak_speed_t3_v2_20260728"
LDLJ_MIN_SAMPLES = 7
LDLJ_EPS = 1e-12

# Trajectory error method: 'path' = total path length, 'excess' = path_length - straight_line_distance
TRAJ_ERROR_METHOD = "excess"
# Planned task comparisons. These are the only task-level comparisons used for
# visualization and inference; all six tasks are never pooled into one model.
TASK_PHASES = {
	"Tapes_Seated": "grasp",
	"Coin_Task_Seated": "grasp",
	"Tapes_Standing": "grasp",
	"Cognitive": "grasp",
	"Pegs_Seated": "right_to_left_grasp",
	"Pegs_Standing": "right_to_left_grasp",
}

TASK_DISPLAY_NAMES = {
	"Tapes_Seated": "Tapes",
	"Coin_Task_Seated": "Coins",
	"Tapes_Standing": "Tapes Standing",
	"Cognitive": "Cognitive",
	"Pegs_Seated": "Pegs",
	"Pegs_Standing": "Pegs Standing",
}

TASK_COLORS = {
	"Tapes_Seated": "#F0E442",
	"Coin_Task_Seated": "#7F7F7F",
	"Tapes_Standing": "#009E73",
	"Cognitive": "#E69F00",
	"Pegs_Seated": "#0072B2",
	"Pegs_Standing": "#8c564b",
}

PLANNED_TASK_COMPARISONS = [
	("tapes_vs_coins", "Tapes_Seated", "Coin_Task_Seated"),
	("tapes_vs_tapes_standing", "Tapes_Seated", "Tapes_Standing"),
	("tapes_vs_cognitive", "Tapes_Seated", "Cognitive"),
	("tapes_vs_pegs", "Tapes_Seated", "Pegs_Seated"),
]


def _interp_fill_1d(x: np.ndarray) -> np.ndarray:
	"""Linear interpolate NaNs for a 1D signal."""
	s = pd.Series(x, dtype=float).interpolate(limit_direction="both")
	return s.to_numpy(dtype=float)


def lowpass_safe(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
	"""Safe low-pass filter with NaN interpolation and short-signal guard."""
	x = np.asarray(x, dtype=float)
	if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
		return x
	if np.isnan(x).any():
		x_f = _interp_fill_1d(x)
		if np.isnan(x_f).any():
			return x
		x = x_f

	b, a = butter(order, cutoff_hz / (fs * 0.5), btype="low")
	padlen = 3 * (max(len(a), len(b)) - 1)
	if x.size <= padlen:
		return x

	try:
		return filtfilt(b, a, x, method="gust")
	except Exception:
		return x


def dimensionless_jerk(speed: np.ndarray, fs: float) -> float:
	"""Compute dimensionless jerk from a non-negative 3D speed profile.

	This follows the published speed-profile implementation: jerk is the second
	temporal derivative of speed and the integrated squared jerk is normalized
	by ``duration**3 / peak_speed**2``. The result is invariant to spatial and
	temporal scaling.
	"""
	movement = np.asarray(speed, dtype=float).reshape(-1)
	if (
		not np.isfinite(fs)
		or fs <= 0
		or movement.size < LDLJ_MIN_SAMPLES
		or not np.all(np.isfinite(movement))
	):
		return np.nan

	dt_local = 1.0 / float(fs)
	duration = (movement.size - 1) * dt_local
	peak_speed = float(np.max(np.abs(movement)))
	if duration <= 0 or peak_speed <= LDLJ_EPS:
		return np.nan

	jerk = np.diff(movement, n=2) / (dt_local ** 2)
	integrated_squared_jerk = float(np.sum(jerk ** 2) * dt_local)
	if (
		not np.isfinite(integrated_squared_jerk)
		or integrated_squared_jerk <= LDLJ_EPS
	):
		return np.nan

	return float(
		-(duration ** 3 / peak_speed ** 2) * integrated_squared_jerk
	)


def log_dimensionless_jerk(speed: np.ndarray, fs: float) -> float:
	"""Return LDLJ, where larger (less negative) values mean smoother motion."""
	dlj = dimensionless_jerk(speed, fs)
	if not np.isfinite(dlj) or dlj == 0:
		return np.nan
	return float(-np.log(abs(dlj)))


def jerk_cost_from_pos(pos_xyz: np.ndarray, fs: float, pos_lp_hz: float = 0.0) -> float:
	"""
	Compute standard speed-profile LDLJ for a 3D position trajectory.

	The filtered 3D position is differentiated once, and LDLJ is calculated from
	the magnitude of the resulting velocity vector. This avoids axis-wise
	averaging and dependence on the absolute coordinate origin.
	Higher LDLJ values indicate smoother movement.
	"""
	X = np.asarray(pos_xyz, dtype=float).copy()
	if X.ndim != 2 or X.shape[1] != 3:
		return np.nan

	if X.shape[0] < LDLJ_MIN_SAMPLES:
		return np.nan

	for k in range(3):
		if np.isnan(X[:, k]).any():
			X[:, k] = _interp_fill_1d(X[:, k])

	if pos_lp_hz > 0:
		for k in range(3):
			X[:, k] = lowpass_safe(X[:, k], pos_lp_hz, fs, order=2)

	velocity_xyz = np.gradient(X, 1.0 / float(fs), axis=0)
	speed = np.linalg.norm(velocity_xyz, axis=1)
	return log_dimensionless_jerk(speed, fs)


def validate_ldlj_implementation() -> None:
	"""Fail fast if the 3D LDLJ loses rigid-transform or scaling invariance."""
	t = np.linspace(0.0, 1.0, 121)
	minimum_jerk = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
	base = np.column_stack(
		[minimum_jerk, 0.35 * minimum_jerk, -0.20 * minimum_jerk]
	)
	angle = np.deg2rad(37.0)
	rotation = np.array(
		[
			[np.cos(angle), -np.sin(angle), 0.0],
			[np.sin(angle), np.cos(angle), 0.0],
			[0.0, 0.0, 1.0],
		]
	)
	reference = jerk_cost_from_pos(base, fs=120.0, pos_lp_hz=0.0)
	checks = {
		"translation": jerk_cost_from_pos(
			base + np.array([1250.0, -330.0, 84.0]),
			fs=120.0,
			pos_lp_hz=0.0,
		),
		"rotation": jerk_cost_from_pos(
			base @ rotation.T,
			fs=120.0,
			pos_lp_hz=0.0,
		),
		"spatial_scale": jerk_cost_from_pos(
			7.5 * base,
			fs=120.0,
			pos_lp_hz=0.0,
		),
		"duration": jerk_cost_from_pos(
			base,
			fs=60.0,
			pos_lp_hz=0.0,
		),
	}
	if not np.isfinite(reference):
		raise RuntimeError("LDLJ self-check failed: non-finite reference value.")
	for name, value in checks.items():
		if not np.isfinite(value) or not np.isclose(
			value, reference, rtol=1e-9, atol=1e-9
		):
			raise RuntimeError(
				f"LDLJ self-check failed for {name}: "
				f"reference={reference}, observed={value}"
			)


def make_segments_four_stage(y: np.ndarray, fs: float) -> List[Tuple[str, int, int]]:
	"""
	Build four-stage movement segments from WRIST_y signal.

	Returns
	-------
	List of (phase, start_idx, end_idx), inclusive.
	"""
	y = np.asarray(y, dtype=float)
	if np.isnan(y).any():
		y = _interp_fill_1d(y)

	y_f = lowpass_safe(y, lowpass_cutoff_hz, fs, lowpass_order) if use_lowpass else y
	prom = max(np.nanstd(y_f) * prominence_factor, 1e-6)
	distance = max(int(fs * min_distance_s), 1)

	peaks, _ = find_peaks(y_f, prominence=prom, distance=distance)
	troughs, _ = find_peaks(-y_f, prominence=prom, distance=distance)

	troughs = np.unique(np.r_[0, troughs, len(y_f) - 1])
	troughs.sort()

	segs: List[Tuple[str, int, int]] = []
	prev_crossing_low_down = None

	for i in range(len(troughs) - 1):
		trough_start = int(troughs[i])
		trough_end = int(troughs[i + 1])

		peaks_in_cycle = peaks[(peaks > trough_start) & (peaks < trough_end)]
		if len(peaks_in_cycle) == 0:
			continue

		peak_idx = int(peaks_in_cycle[np.argmax(y_f[peaks_in_cycle])])
		peak_value = y_f[peak_idx]

		if peak_value < THRESHOLD_HIGH:
			continue

		segment_up = y_f[trough_start:peak_idx + 1]
		crossing_low_up = None
		for j, val in enumerate(segment_up):
			if val >= THRESHOLD_LOW:
				crossing_low_up = trough_start + j
				break

		if crossing_low_up is None:
			continue

		segment_carry = y_f[crossing_low_up:peak_idx + 1]
		crossing_high_up = None
		for j, val in enumerate(segment_carry):
			if val >= THRESHOLD_HIGH:
				crossing_high_up = crossing_low_up + j
				break

		if crossing_high_up is None:
			continue

		segment_down_high = y_f[peak_idx:trough_end + 1]
		crossing_high_down = None
		for j, val in enumerate(segment_down_high):
			if val <= THRESHOLD_HIGH:
				crossing_high_down = peak_idx + j
				break

		if crossing_high_down is None:
			continue

		segment_down_low = y_f[crossing_high_down:trough_end + 1]
		crossing_low_down = None
		for j, val in enumerate(segment_down_low):
			if val <= THRESHOLD_LOW:
				crossing_low_down = crossing_high_down + j
				break

		if crossing_low_down is None:
			continue

		grasp_start = prev_crossing_low_down if prev_crossing_low_down is not None else trough_start

		if crossing_low_up - grasp_start + 1 >= min_seg_len_frames:
			segs.append(("grasp", grasp_start, crossing_low_up))
		if crossing_high_up - crossing_low_up + 1 >= min_seg_len_frames:
			segs.append(("carrying", crossing_low_up, crossing_high_up))
		if peak_idx - crossing_high_up + 1 >= min_seg_len_frames:
			segs.append(("placement", crossing_high_up, peak_idx))
		if crossing_low_down - peak_idx + 1 >= min_seg_len_frames:
			segs.append(("return", peak_idx, crossing_low_down))

		prev_crossing_low_down = crossing_low_down

	first_grasp_removed = False
	segs_filtered: List[Tuple[str, int, int]] = []
	for phase, start, end in segs:
		if phase == "grasp" and not first_grasp_removed:
			first_grasp_removed = True
			continue
		segs_filtered.append((phase, start, end))

	return segs_filtered


def make_task_segments_for_block(
	task_key: str,
	subject: str,
	day: int,
	block: int,
	df: pd.DataFrame,
	fs: float,
) -> List[Tuple[str, int, int]]:
	"""Return default analysis segments with task-specific phase labels."""
	segments = _common_make_task_segments(task_key, subject, day, block, df, fs=fs)
	phase = "right_to_left_grasp" if task_key.startswith("Pegs_") else "grasp"
	return [(phase, int(a), int(b)) for a, b in segments]


def read_block_csv(subject: str, day: int, block: int, task_name: str) -> Path:
	"""Path to absref CSV for a block."""
	subj_day = f"{subject}_Day{day}"
	trial = f"{task_name}_{block}"
	return absref_root / subject / subj_day / trial / f"{trial}_with_table_absref.csv"


def read_task_block_csv(subject: str, day: int, block: int, task_key: str) -> Path:
	return _shared_trial_csv_path(absref_root, subject, day, block, task_key)


def task_name_candidates(task_key: str, subject: str) -> List[str]:
	return _shared_task_name_candidates(task_key, subject)


def resolve_task_name(task_key: str, subject: str) -> str:
	"""Resolve preferred per-subject task folder/file prefix used in absref data."""
	return task_name_candidates(task_key, subject)[0]


def load_marker_3d(df: pd.DataFrame, marker: str) -> Optional[np.ndarray]:
	"""Extract 3D position from dataframe for a marker (X, Y, Z columns)."""
	cols_needed = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
	if not all(c in df.columns for c in cols_needed):
		return None
	return df[cols_needed].to_numpy(dtype=float)


def per_subject_block_marker_jerk(
	subject: str,
	day: int,
	block: int,
	task_key: str,
	marker: str,
	phase: str = "all"
) -> float:
	"""Compute mean jerk (LDLJ) for a marker in a block."""
	csv_path = read_task_block_csv(subject, day, block, task_key)
	if not csv_path.exists():
		return np.nan

	try:
		df = pd.read_csv(csv_path)
	except Exception:
		return np.nan

	if marker_seg not in df.columns:
		return np.nan

	y = df[marker_seg].to_numpy(dtype=float)
	segs = make_task_segments_for_block(task_key, subject, day, block, df, fs)
	if len(segs) == 0:
		return np.nan

	pos_all = load_marker_3d(df, marker)
	if pos_all is None or pos_all.shape[0] == 0:
		return np.nan

	jerk_vals: List[float] = []
	for seg_phase, a, b in segs:
		if phase != "all" and seg_phase != phase:
			continue
		if b >= pos_all.shape[0]:
			b = pos_all.shape[0] - 1
		if b - a + 1 < min_seg_len_frames:
			continue

		pos_seg = pos_all[a:b + 1, :]
		jerk_val = jerk_cost_from_pos(pos_seg, fs=fs, pos_lp_hz=jerk_pos_lowpass_hz)
		if np.isfinite(jerk_val):
			jerk_vals.append(jerk_val)

	if len(jerk_vals) == 0:
		return np.nan
	return float(np.mean(jerk_vals))


def aggregate_jerk_data(task_key: str, marker: str, phase: str = "all") -> pd.DataFrame:
	"""Aggregate jerk data across subjects, days, blocks."""
	rows = []
	available_days = TASK_AVAILABLE_DAYS.get(task_key)
	task_days = [day for day in day_list if available_days is None or day in available_days]
	for day in task_days:
		for block in block_list:
			x = (day - 1) * 5 + block
			for subj in subject_list:
				val = per_subject_block_marker_jerk(subj, day, block, task_key, marker, phase)
				if np.isfinite(val):
					rows.append({
						"Subject": subj,
						"Day": day,
						"Block": block,
						"x": x,
						"metric_value": float(val),
					})
	return pd.DataFrame(rows)


def summarize_metric_blocks(df: pd.DataFrame, stat_method: str = "mean_sem") -> pd.DataFrame:
	"""Compute population statistics per block for metric_value."""
	if df.empty:
		return pd.DataFrame()

	g = df.groupby(["Day", "Block", "x"])["metric_value"]
	rows = []
	for (day, block, x), group_vals in g:
		vals = group_vals.to_numpy(dtype=float)
		vals = vals[np.isfinite(vals)]
		if len(vals) == 0:
			continue

		if stat_method == "median_se":
			center = np.median(vals)
			if len(vals) < 2:
				se_median = 0.0
			else:
				sd = np.std(vals, ddof=1)
				se_median = 1.253 * sd / np.sqrt(len(vals))
			lower = center - se_median
			upper = center + se_median
		elif stat_method == "mean_sd":
			center = np.mean(vals)
			sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
			lower = center - sd
			upper = center + sd
		elif stat_method == "mean_sem":
			center = np.mean(vals)
			sem = stats.sem(vals) if len(vals) > 1 else 0.0
			lower = center - sem
			upper = center + sem
		elif stat_method == "mean_ci":
			center = np.mean(vals)
			if len(vals) > 1:
				lower, upper = stats.t.interval(0.95, len(vals) - 1, loc=center, scale=stats.sem(vals))
			else:
				lower = upper = center
		else:
			raise ValueError(f"Unknown stat_method: {stat_method}")

		rows.append({
			"Day": day,
			"Block": block,
			"x": x,
			"center": float(center),
			"lower": float(lower),
			"upper": float(upper),
		})

	return pd.DataFrame(rows).sort_values(["Day", "Block"]).reset_index(drop=True)


def aggregate_day_level_within_subject(
	df_block_level: pd.DataFrame,
	within_method: str = "median"
) -> pd.DataFrame:
	"""
	Aggregate block-level values to day-level within each participant.

	For each Subject x Day, compute either median or mean across available blocks.
	"""
	if df_block_level.empty:
		return pd.DataFrame(columns=["Subject", "Day", "x", "metric_value"])

	if within_method not in {"median", "mean"}:
		raise ValueError(f"Unknown within_method: {within_method}")

	rows = []
	g = df_block_level.groupby(["Subject", "Day"]) ["metric_value"]
	for (subj, day), group_vals in g:
		vals = group_vals.to_numpy(dtype=float)
		vals = vals[np.isfinite(vals)]
		if len(vals) == 0:
			continue

		if within_method == "median":
			v = float(np.median(vals))
		else:
			v = float(np.mean(vals))

		rows.append({
			"Subject": subj,
			"Day": int(day),
			"x": int(day),
			"metric_value": v,
		})

	return pd.DataFrame(rows).sort_values(["Subject", "Day"]).reset_index(drop=True)


def summarize_metric_days(df: pd.DataFrame, stat_method: str = "mean_sem") -> pd.DataFrame:
	"""Compute population statistics per day for metric_value."""
	if df.empty:
		return pd.DataFrame()

	g = df.groupby(["Day", "x"]) ["metric_value"]
	rows = []
	for (day, x), group_vals in g:
		vals = group_vals.to_numpy(dtype=float)
		vals = vals[np.isfinite(vals)]
		if len(vals) == 0:
			continue

		if stat_method == "median_se":
			center = np.median(vals)
			if len(vals) < 2:
				se_median = 0.0
			else:
				sd = np.std(vals, ddof=1)
				se_median = 1.253 * sd / np.sqrt(len(vals))
			lower = center - se_median
			upper = center + se_median
		elif stat_method == "mean_sd":
			center = np.mean(vals)
			sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
			lower = center - sd
			upper = center + sd
		elif stat_method == "mean_sem":
			center = np.mean(vals)
			sem = stats.sem(vals) if len(vals) > 1 else 0.0
			lower = center - sem
			upper = center + sem
		elif stat_method == "mean_ci":
			center = np.mean(vals)
			if len(vals) > 1:
				lower, upper = stats.t.interval(0.95, len(vals) - 1, loc=center, scale=stats.sem(vals))
			else:
				lower = upper = center
		else:
			raise ValueError(f"Unknown stat_method: {stat_method}")

		rows.append({
			"Day": int(day),
			"x": int(x),
			"center": float(center),
			"lower": float(lower),
			"upper": float(upper),
		})

	return pd.DataFrame(rows).sort_values(["Day"]).reset_index(drop=True)


def plot_metric_bar_errorbar(
	task_key: str,
	marker: str,
	phase: str,
	df_individual: pd.DataFrame,
	df_summary: pd.DataFrame,
	stat_method: str = "mean_sem"
) -> tuple:
	"""Plot jerk as bars with error bars and individual participant points."""
	fig, ax = plt.subplots(figsize=(12, 9))
	rng = np.random.default_rng(123)

	if df_summary.empty:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		return fig, ax

	x_pos = df_summary["x"].to_numpy()
	center = df_summary["center"].to_numpy()
	lower = df_summary["lower"].to_numpy()
	upper = df_summary["upper"].to_numpy()

	err_lower = center - lower
	err_upper = upper - center

	stat_labels = {
		"median_se": "Median +/- SE",
		"mean_sd": "Mean ± SD",
		"mean_sem": "Mean ± SEM",
		"mean_ci": "Mean ± 95% CI",
	}
	stat_label = stat_labels.get(stat_method, stat_method)

	ax.bar(
		x_pos,
		center,
		yerr=[err_lower, err_upper],
		width=0.95,
		alpha=0.7,
		color="skyblue",
		edgecolor="black",
		linewidth=0.8,
		capsize=3,
		label=f"Population ({stat_label})",
	)

	for subj in subject_list:
		sdf = df_individual[df_individual["Subject"] == subj]
		if not sdf.empty:
			ax.scatter(
				sdf["x"] + rng.normal(0, 0.02, size=len(sdf)),
				sdf["metric_value"],
				s=20,
				alpha=0.6,
				color="gray",
				edgecolor="white",
				linewidth=0.3,
			)

	xticks = [1, 6, 11, 16, 21]
	ax.set_xticks(xticks)
	ax.set_xticklabels([f"Day{d}" for d in range(1, 6)])

	for d in range(2, 6):
		ax.axvline((d - 1) * 5 + 0.5, linestyle="--", linewidth=0.8, alpha=0.3)

	ax.set_xlim(0.5, 25.5)
	ax.set_xlabel("Training Day", fontsize=11)
	ax.set_ylabel("LDLJ (Log Dimensionless Jerk)", fontsize=11)
	phase_label = f" ({phase.capitalize()})" if phase != "all" else ""
	ax.set_title(f"{marker} LDLJ{phase_label} - {task_key} ({stat_method})")
	ax.grid(True, alpha=0.2, axis="y")

	return fig, ax


def plot_metric_day_bar_errorbar(
	task_key: str,
	marker: str,
	phase: str,
	within_method: str,
	df_individual_day: pd.DataFrame,
	df_summary_day: pd.DataFrame,
	stat_method: str = "mean_sem"
) -> tuple:
	"""Plot day-level jerk bars with participant points."""
	fig, ax = plt.subplots(figsize=(10, 8))
	rng = np.random.default_rng(321)

	if df_summary_day.empty:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		return fig, ax

	x_pos = df_summary_day["x"].to_numpy()
	center = df_summary_day["center"].to_numpy()
	lower = df_summary_day["lower"].to_numpy()
	upper = df_summary_day["upper"].to_numpy()

	err_lower = center - lower
	err_upper = upper - center

	stat_labels = {
		"median_se": "Median +/- SE",
		"mean_sd": "Mean ± SD",
		"mean_sem": "Mean ± SEM",
		"mean_ci": "Mean ± 95% CI",
	}
	stat_label = stat_labels.get(stat_method, stat_method)

	ax.bar(
		x_pos,
		center,
		yerr=[err_lower, err_upper],
		width=0.8,
		alpha=0.7,
		color="lightcoral",
		edgecolor="black",
		linewidth=0.8,
		capsize=3,
		label=f"Population ({stat_label})",
	)

	for subj in subject_list:
		sdf = df_individual_day[df_individual_day["Subject"] == subj]
		if not sdf.empty:
			ax.scatter(
				sdf["x"] + rng.normal(0, 0.02, size=len(sdf)),
				sdf["metric_value"],
				s=24,
				alpha=0.65,
				color="gray",
				edgecolor="white",
				linewidth=0.3,
			)

	ax.set_xticks(day_list)
	ax.set_xticklabels([f"Day{d}" for d in day_list])
	ax.set_xlim(min(day_list) - 0.5, max(day_list) + 0.5)
	ax.set_xlabel("Training Day", fontsize=11)
	ax.set_ylabel("LDLJ (Log Dimensionless Jerk)", fontsize=11)
	phase_label = f" ({phase.capitalize()})" if phase != "all" else ""
	ax.set_title(
		f"{marker} LDLJ{phase_label} - {task_key} Day-level "
		f"(within-subject {within_method}, {stat_method})"
	)
	ax.grid(True, alpha=0.2, axis="y")

	return fig, ax


def method_label(method: str) -> str:
	"""Human-readable label for summary/error-bar method."""
	labels = {
		"median_se": "Median +/- SE",
		"mean_sd": "Mean +/- SD",
		"mean_sem": "Mean +/- SEM",
		"mean_ci": "Mean +/- 95% CI",
	}
	return labels.get(method, method)


def comparison_title(task_pair: Tuple[str, str]) -> str:
	"""Human-readable title for a two-task comparison."""
	return " versus ".join(TASK_DISPLAY_NAMES.get(task, task) for task in task_pair)


def task_file_label(task: str) -> str:
	return str(task).replace(" ", "_").replace("/", "-")


def plot_ldlj_pair(
	summary_day_by_key: Dict[Tuple[str, str, str, str, str], pd.DataFrame],
	within_method: str,
	stat_method: str,
	out_dir: Path,
	comparison_name: str,
	task_pair: Tuple[str, str],
) -> None:
	"""Plot LDLJ in the same group-level multi-panel style as smoothness-SPARC.py."""
	fig, axes = plt.subplots(1, len(MARKERS), figsize=(5 * len(MARKERS), 4.7), sharey=True)
	if len(MARKERS) == 1:
		axes = [axes]

	for axis, marker in zip(axes, MARKERS):
		comparison_days = set(day_list)
		for task_key in task_pair:
			phase = TASK_PHASES[task_key]
			frame = summary_day_by_key.get((task_key, marker, phase, within_method, stat_method), pd.DataFrame())
			if not frame.empty:
				comparison_days &= set(frame["Day"].astype(int).tolist())
		comparison_days = sorted(comparison_days)

		for task_key in task_pair:
			phase = TASK_PHASES[task_key]
			frame = summary_day_by_key.get((task_key, marker, phase, within_method, stat_method), pd.DataFrame())
			if frame.empty:
				continue
			if comparison_days:
				frame = frame[frame["Day"].isin(comparison_days)].copy()
			frame = frame.sort_values("Day")
			if frame.empty:
				continue

			day = frame["Day"].to_numpy(dtype=float)
			center = frame["center"].to_numpy(dtype=float)
			color = TASK_COLORS.get(task_key, "#999999")
			axis.plot(
				day,
				center,
				marker="o",
				linewidth=2,
				color=color,
				label=TASK_DISPLAY_NAMES.get(task_key, task_key),
			)
			axis.fill_between(
				day,
				frame["lower"].to_numpy(dtype=float),
				frame["upper"].to_numpy(dtype=float),
				color=color,
				alpha=0.2,
			)

		axis.set_title(marker)
		axis.set_xlabel("Training day")
		axis.set_xticks(comparison_days if comparison_days else sorted(day_list))
		axis.grid(True, axis="y", alpha=0.25)
		axis.spines["top"].set_visible(False)
		axis.spines["right"].set_visible(False)

	axes[0].set_ylabel("LDLJ (higher = smoother)")
	axes[0].legend(frameon=False)
	fig.suptitle(
		f"Grasp LDLJ: {comparison_title(task_pair)} | "
		f"within-subject {within_method} | {method_label(stat_method)}"
	)
	fig.tight_layout()
	fig.savefig(
		out_dir / f"ldlj_grasp_group_day_{comparison_name}_within_{within_method}_{stat_method}.png",
		dpi=300,
		bbox_inches="tight",
	)
	plt.close(fig)


def plot_ldlj_single_task(
	summary_day_by_key: Dict[Tuple[str, str, str, str, str], pd.DataFrame],
	within_method: str,
	stat_method: str,
	out_dir: Path,
	task_key: str,
) -> None:
	phase = TASK_PHASES[task_key]
	fig, axes = plt.subplots(1, len(MARKERS), figsize=(5 * len(MARKERS), 4.7), sharey=True)
	if len(MARKERS) == 1:
		axes = [axes]

	for axis, marker in zip(axes, MARKERS):
		frame = summary_day_by_key.get((task_key, marker, phase, within_method, stat_method), pd.DataFrame())
		if frame.empty:
			continue
		frame = frame.sort_values("Day")
		day = frame["Day"].to_numpy(dtype=float)
		center = frame["center"].to_numpy(dtype=float)
		color = TASK_COLORS.get(task_key, "#999999")
		axis.plot(day, center, marker="o", linewidth=2, color=color, label=TASK_DISPLAY_NAMES.get(task_key, task_key))
		axis.fill_between(day, frame["lower"].to_numpy(dtype=float), frame["upper"].to_numpy(dtype=float), color=color, alpha=0.2)
		axis.set_title(marker)
		axis.set_xlabel("Training day")
		axis.set_xticks(sorted(frame["Day"].astype(int).unique().tolist()))
		axis.grid(True, axis="y", alpha=0.25)
		axis.spines["top"].set_visible(False)
		axis.spines["right"].set_visible(False)

	axes[0].set_ylabel("LDLJ (higher = smoother)")
	axes[0].legend(frameon=False)
	fig.suptitle(
		f"Grasp LDLJ: {TASK_DISPLAY_NAMES.get(task_key, task_key)} | "
		f"within-subject {within_method} | {method_label(stat_method)}"
	)
	fig.tight_layout()
	task_dir = out_dir / "per_task" / task_file_label(task_key)
	task_dir.mkdir(parents=True, exist_ok=True)
	fig.savefig(
		task_dir / f"ldlj_grasp_group_day_{task_file_label(task_key)}_within_{within_method}_{stat_method}.png",
		dpi=300,
		bbox_inches="tight",
	)
	plt.close(fig)


def fit_lmm(df: pd.DataFrame, formula: str, groups_col: str, re_formula: Optional[str] = None):
	"""Fit LMM (REML); tries multiple optimizers for robustness."""
	kw = {"re_formula": re_formula} if re_formula else {}
	methods = ["powell", "nm", "bfgs", "cg"]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", ConvergenceWarning)
		warnings.simplefilter("ignore", UserWarning)
		model = smf.mixedlm(formula, data=df, groups=df[groups_col], **kw)
		last_exc = None
		for method in methods:
			try:
				return model.fit(reml=True, method=method, maxiter=2000, disp=False)
			except Exception as e:
				last_exc = e
		raise RuntimeError(f"All optimizers failed. Last error: {last_exc}")


def lmm_result_to_table(result, metadata: Dict[str, object]) -> pd.DataFrame:
	"""Convert LMM fixed effects to a tidy table with metadata."""
	fe_names = result.fe_params.index
	beta = result.fe_params
	se = result.bse[fe_names]
	pv = result.pvalues[fe_names]
	rows = []
	for term in fe_names:
		rows.append({
			**metadata,
			"term": term,
			"beta": float(beta[term]),
			"SE": float(se[term]),
			"CI_lower": float(beta[term] - 1.96 * se[term]),
			"CI_upper": float(beta[term] + 1.96 * se[term]),
			"p_value": float(pv[term]),
			"n_subjects": int(getattr(result, "ngroups", result.model.n_groups)),
			"n_observations": int(result.nobs),
			"log_likelihood": float(result.llf),
		})
	return pd.DataFrame(rows)


def print_lmm_summary(result, title: str) -> None:
	"""Print compact fixed-effect LMM summary."""
	sep = "=" * 80
	print(f"\n{sep}")
	print(f"  {title}")
	print(sep)
	fe_names = result.fe_params.index
	beta = result.fe_params
	se = result.bse[fe_names]
	pv = result.pvalues[fe_names]
	print(f"  {'Term':<35} {'beta':>9} {'SE':>8} {'95% CI':>18} {'p':>9}  Sig")
	print("  " + "-" * 76)
	for name in fe_names:
		stars = "***" if pv[name] < 0.001 else "**" if pv[name] < 0.01 else "*" if pv[name] < 0.05 else "n.s."
		ci_str = f"[{beta[name] - 1.96 * se[name]:+.3f}, {beta[name] + 1.96 * se[name]:+.3f}]"
		print(f"  {name:<35} {beta[name]:+9.4f} {se[name]:8.4f} {ci_str:>18} {pv[name]:9.4f}  {stars}")
	print(f"\n  Log-likelihood: {result.llf:.4f}")
	print(f"  N groups (subjects): {getattr(result, 'ngroups', result.model.n_groups)}")
	print(f"  N observations: {int(result.nobs)}")
	print(sep + "\n")


def build_pair_lmm_dataframe(task_a: str, task_b: str, marker: str, within_method: str, day_data_by_key: Dict[Tuple[str, str, str, str], pd.DataFrame]) -> pd.DataFrame:
	"""Build one two-task day-level dataset for direct pairwise LMM."""
	frames = []
	for task_key in (task_a, task_b):
		phase = TASK_PHASES[task_key]
		df = day_data_by_key.get((task_key, marker, phase, within_method), pd.DataFrame())
		if df.empty:
			continue
		df = df.copy()
		df["task"] = task_key
		df["phase"] = phase
		frames.append(df)
	if len(frames) < 2:
		return pd.DataFrame()

	combined = pd.concat(frames, ignore_index=True)
	combined = combined.dropna(subset=["metric_value"])
	common_days = sorted(set(combined.loc[combined["task"] == task_a, "Day"]) & set(combined.loc[combined["task"] == task_b, "Day"]))
	if not common_days:
		return pd.DataFrame()
	combined = combined[combined["Day"].isin(common_days)].copy()
	# Tuple order is comparison task, reference task.  Code Tapes/comparison as
	# 1 so coefficients consistently represent comparison minus reference.
	combined["task_code"] = (combined["task"] == task_a).astype(float)
	combined["day_c"] = combined["Day"].astype(float) - float(min(common_days))
	return combined


def run_pairwise_lmm_analysis(day_data_by_key: Dict[Tuple[str, str, str, str], pd.DataFrame]) -> pd.DataFrame:
	"""Run direct Task x Day LMM for each planned two-task comparison."""
	print("\n" + "=" * 80)
	print("Pairwise LMM - Smoothness (LDLJ), direct planned task comparisons")
	print("=" * 80)
	all_tables = []

	for comparison_name, task_a, task_b in PLANNED_TASK_COMPARISONS:
		comparison_dir = OUT_DIR / comparison_name
		comparison_dir.mkdir(parents=True, exist_ok=True)
		for marker in MARKERS:
			for within_method in DAY_LEVEL_WITHIN_METHODS:
				combined = build_pair_lmm_dataframe(task_a, task_b, marker, within_method, day_data_by_key)
				if combined.empty or combined["task"].nunique() < 2:
					print(f"  [skip] No paired task data: {comparison_name} {marker} within={within_method}")
					continue

				desc = combined.groupby(["task", "phase", "Day"])["metric_value"].agg(["count", "mean", "std"]).rename(columns={"count": "n", "mean": "mean", "std": "SD"}).reset_index()
				desc_csv = comparison_dir / f"{comparison_name}_{marker}_within_{within_method}_descriptive.csv"
				desc.to_csv(desc_csv, index=False, float_format="%.6f")
				pair_csv = comparison_dir / f"{comparison_name}_{marker}_within_{within_method}_individual_for_lmm.csv"
				combined.to_csv(pair_csv, index=False, float_format="%.6f")

				label = f"{comparison_name} | {marker} | within={within_method}"
				print(f"\n{label}")
				print(desc.round(4))
				fname_base = f"{comparison_name}_{marker}_within_{within_method}"

				for model_name, re_formula in (("intercept_only", None), ("random_slope", "~day_c")):
					try:
						result = fit_lmm(combined, "metric_value ~ task_code * day_c", "Subject", re_formula=re_formula)
					except Exception as e:
						level = "warn" if model_name == "random_slope" else "error"
						print(f"  [{level}] {model_name} failed: {e}")
						if model_name == "intercept_only":
							break
						continue

					print_lmm_summary(result, f"LDLJ {label} - {model_name}")
					metadata = {
						"comparison": comparison_name,
						"comparison_task": task_a,
						"reference_task": task_b,
						"comparison_phase": TASK_PHASES[task_a],
						"reference_phase": TASK_PHASES[task_b],
						"marker": marker,
						"within_method": within_method,
						"model": model_name,
					}
					table = lmm_result_to_table(result, metadata)
					out_csv = comparison_dir / f"{fname_base}_lmm_{model_name}.csv"
					table.to_csv(out_csv, index=False, float_format="%.6f")
					print(f"  Saved: {out_csv.name}")
					all_tables.append(table)

	if not all_tables:
		return pd.DataFrame()

	all_results = pd.concat(all_tables, ignore_index=True)
	all_csv = OUT_DIR / "all_pairwise_lmm_fixed_effects.csv"
	all_results.to_csv(all_csv, index=False, float_format="%.6f")
	print(f"\nSaved: {all_csv.name}")

	task_terms = all_results[all_results["term"].str.contains("task_code", regex=False)].copy()
	if not task_terms.empty:
		task_terms["p_value_fdr_bh"] = np.nan
		for _, idx in task_terms.groupby(["marker", "within_method", "model", "term"]).groups.items():
			pvals = task_terms.loc[idx, "p_value"].to_numpy(dtype=float)
			_, p_corr, _, _ = multipletests(pvals, method="fdr_bh")
			task_terms.loc[idx, "p_value_fdr_bh"] = p_corr
		fdr_csv = OUT_DIR / "all_pairwise_lmm_task_terms_fdr_bh.csv"
		task_terms.to_csv(fdr_csv, index=False, float_format="%.6f")
		print(f"Saved: {fdr_csv.name}")
	return all_results


def main() -> None:
	"""Standalone entry point for LDLJ computation, pairwise plots, and pairwise LMM."""
	validate_ldlj_implementation()
	method_metadata = {
		"implementation_version": LDLJ_IMPLEMENTATION_VERSION,
		"input_signal": "3D marker position converted to speed magnitude",
		"derivative_order": "second derivative of speed (third derivative of position)",
		"normalization": "duration^3 divided by squared peak speed",
		"aggregation_across_axes": "Euclidean magnitude of the 3D velocity vector",
		"position_lowpass_hz": jerk_pos_lowpass_hz,
		"sampling_frequency_hz": fs,
		"higher_values_mean": "smoother movement",
		"independent_statistical_unit": "participant",
	}
	metadata_path = OUT_DIR / "ldlj_method_metadata.json"
	metadata_path.write_text(
		json.dumps(method_metadata, indent=2),
		encoding="utf-8",
	)
	print("\n" + "=" * 80)
	print("Smoothness (LDLJ) Analysis - Planned pairwise task comparisons")
	print("=" * 80)
	print(f"LDLJ implementation: {LDLJ_IMPLEMENTATION_VERSION}")
	print(f"Saved: {metadata_path.name}")

	day_data_by_key: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
	summary_day_by_key: Dict[Tuple[str, str, str, str, str], pd.DataFrame] = {}

	for task_key in SELECTED_TASKS:
		if task_key not in TASKS:
			print(f"Warning: Task '{task_key}' not found in TASKS dict, skipping.")
			continue
		task_output_dir = OUT_DIR / "task_level_csv" / task_key
		task_output_dir.mkdir(parents=True, exist_ok=True)
		phase = TASK_PHASES.get(task_key, "grasp")
		print(f"\nProcessing task: {task_key} | phase: {phase}")

		for marker in MARKERS:
			df_individual_block = aggregate_jerk_data(task_key, marker, phase)
			if df_individual_block.empty:
				print(f"  No valid data for {marker}, skip")
				continue
			individual_csv = task_output_dir / f"{task_key}_{marker}_jerk_{phase}_individual.csv"
			df_individual_block.to_csv(individual_csv, index=False)
			print(f"  Saved: {individual_csv.name}")

			for stat_method in STATISTICAL_METHODS:
				df_summary = summarize_metric_blocks(df_individual_block, stat_method)
				if not df_summary.empty:
					summary_csv = task_output_dir / f"{task_key}_{marker}_jerk_{phase}_{stat_method}_summary.csv"
					df_summary.to_csv(summary_csv, index=False)

			for within_method in DAY_LEVEL_WITHIN_METHODS:
				df_individual_day = aggregate_day_level_within_subject(df_individual_block, within_method=within_method)
				if df_individual_day.empty:
					continue
				day_data_by_key[(task_key, marker, phase, within_method)] = df_individual_day
				day_individual_csv = task_output_dir / f"{task_key}_{marker}_jerk_{phase}_day_level_within_{within_method}_individual.csv"
				df_individual_day.to_csv(day_individual_csv, index=False)
				print(f"  Saved: {day_individual_csv.name}")

				for stat_method in STATISTICAL_METHODS:
					df_summary_day = summarize_metric_days(df_individual_day, stat_method)
					if df_summary_day.empty:
						continue
					summary_day_by_key[(task_key, marker, phase, within_method, stat_method)] = df_summary_day
					day_summary_csv = task_output_dir / f"{task_key}_{marker}_jerk_{phase}_day_level_within_{within_method}_{stat_method}_summary.csv"
					df_summary_day.to_csv(day_summary_csv, index=False)

	print("\n" + "=" * 80)
	print("Generating planned pairwise SPARC-style line plots")
	print("=" * 80)
	for comparison_name, task_a, task_b in PLANNED_TASK_COMPARISONS:
		comparison_dir = OUT_DIR / comparison_name
		comparison_dir.mkdir(parents=True, exist_ok=True)
		print(f"\nComparison: {comparison_name}")
		for within_method in DAY_LEVEL_WITHIN_METHODS:
			for stat_method in STATISTICAL_METHODS:
				plot_ldlj_pair(
					summary_day_by_key=summary_day_by_key,
					within_method=within_method,
					stat_method=stat_method,
					out_dir=comparison_dir,
					comparison_name=comparison_name,
					task_pair=(task_a, task_b),
				)
				print(
					"  Saved: "
					f"ldlj_grasp_group_day_{comparison_name}_within_{within_method}_{stat_method}.png"
				)

	print("\n" + "=" * 80)
	print("Generating single-task LDLJ line plots")
	print("=" * 80)
	for task_key in SELECTED_TASKS:
		for within_method in DAY_LEVEL_WITHIN_METHODS:
			for stat_method in STATISTICAL_METHODS:
				plot_ldlj_single_task(
					summary_day_by_key=summary_day_by_key,
					within_method=within_method,
					stat_method=stat_method,
					out_dir=OUT_DIR,
					task_key=task_key,
				)

	run_pairwise_lmm_analysis(day_data_by_key)
	print("\n" + "=" * 80)
	print(f"Analysis complete. Results saved to: {OUT_DIR}")
	print("Statistical note: LMMs are direct two-task planned comparisons only; task-related p-values are also saved with BH FDR correction across the four planned comparisons for each term/model.")
	print("=" * 80 + "\n")


if __name__ == "__main__":
	main()
