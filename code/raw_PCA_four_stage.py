#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raw-angle PCA analysis with canonical grasp segmentation.

This script keeps the same:
- canonical grasp segmentation from D_kinematic/grasp_kinematic_common.py,
- traversal granularity (subject/task/day/block),
- within-subject analysis,
- angle-group definitions.

For each (subject, task, day, block, phase, angle_group), all normalized
segments are concatenated along time and projected to a fixed PCA basis.

Fixed PCA rule:
- For each (subject, task, angle_group), train PCA once using Day1 all blocks
	and all phases combined.
- Apply the same Day1 PCA to Day1-5 (no retraining), keeping within-subject
	and within-task analysis.

Outputs focus on explained-variance ratio (EVR) and cumulative EVR so we can
track how principal-component proportion changes across days.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, filtfilt


KINEMATIC_COMMON_DIR = Path(__file__).resolve().parents[1] / "D_kinematic"
if str(KINEMATIC_COMMON_DIR) not in sys.path:
	sys.path.insert(0, str(KINEMATIC_COMMON_DIR))
RESULT_ANALYSIS_DIR = KINEMATIC_COMMON_DIR.parent
if str(RESULT_ANALYSIS_DIR) not in sys.path:
	sys.path.insert(0, str(RESULT_ANALYSIS_DIR))

from grasp_kinematic_common import make_task_segments as make_common_task_segments
from task_naming import first_existing_trial_paths, task_name_candidates


# =========================
# Configuration
# =========================
DATA_ROOT = Path(
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/NullSpace_Data"
)
ABSREF_ROOT = DATA_ROOT / "Checked_Absref_data"
ANGLE_ROOT = DATA_ROOT / "Joint_angles_time_series"
RESULTS_ROOT = Path(
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/Results_2/raw_pca_four_stage_upper_limb_fingers"
)
DATA_OUT_DIR = RESULTS_ROOT / "data"
VIS_OUT_DIR = RESULTS_ROOT / "figures"

SUBJECT_LIST = [
	"PT_06", "PT_07", "PT_08", "PT_09", "PT_10", "PT_11", "PT_12", "PT_13", "PT_14",
	"PT_15", "PT_16", "PT_17", "PT_18", "PT_19", "PT_20", "PT_21", "PT_22", "PT_23",
	"PT_24", "PT_25", "PT_26",
]

# Subjects where seated-task trials are stored under *_First naming.
SUBJECTS_WITH_FIRST_NAMING = {"PT_23", "PT_24", "PT_25", "PT_26"}

DAY_LIST = [1, 2, 3, 4, 5]
TASK_LIST = ["Tapes", "Tapes_Standing", "Pegs", "Pegs_Standing", "Coins", "Cognitive"]
#TASK_LIST = [ "Cognitive"]
TASK_AVAILABLE_DAYS = {"Cognitive": [1, 3, 5]}
TASK_DISPLAY_NAMES = {
	"Tapes": "Tapes seated",
	"Tapes_Standing": "Tapes standing",
	"Pegs": "Pegs seated",
	"Pegs_Standing": "Pegs standing",
	"Coins": "Coins",
	"Cognitive": "Cognitive",
}
PLANNED_TASK_PAIRS = [
	("Tapes", "Coins"),
	("Tapes", "Tapes_Standing"),
	("Tapes", "Cognitive"),
	("Tapes", "Pegs"),
]
TASK_COLORS = {
	"Tapes": "#F0E442",
	"Tapes_Standing": "#009E73",
	"Pegs": "#0072B2",
	"Pegs_Standing": "#7570b3",
	"Coins": "#7F7F7F",
	"Cognitive": "#E69F00",
}


def task_label(task: str) -> str:
	return TASK_DISPLAY_NAMES.get(task, task)


def comparison_file_label(pair: Tuple[str, str]) -> str:
	return "_vs_".join(task.lower().replace("_", "-") for task in pair)

def pair_has_both_tasks(df: pd.DataFrame, pair: Tuple[str, str]) -> bool:
	present_tasks = set(df["task"].dropna().unique().tolist())
	return all(task in present_tasks for task in pair)



def available_days_for_task(task: str):
	return TASK_AVAILABLE_DAYS.get(task, DAY_LIST)
BLOCK_LIST = [1, 2, 3, 4, 5]

# Two pre-specified anatomical PCA families.
INTEREST_ANGLE_COLUMNS_UPPER_LIMB = [
	"wrist_flex_ext_deg180",
	"wrist_dev_deg",
	"wrist_prosup_deg",
	"elbow_flex_deg",
	"shoulder_elev_deg",
	"shoulder_plane_deg",
	"shoulder_axial_proxy_deg",
]

INTEREST_ANGLE_COLUMNS_FINGERS = [
	"INDEX_mcp_flex_deg",
	"INDEX_pip_flex_deg",
	"INDEX_dip_flex_deg",
	"MIDDLE_mcp_flex_deg",
	"MIDDLE_pip_flex_deg",
	"MIDDLE_dip_flex_deg",
	"RING_mcp_flex_deg",
	"RING_pip_flex_deg",
	"RING_dip_flex_deg",
	"PINKY_mcp_flex_deg",
	"PINKY_pip_flex_deg",
	"PINKY_dip_flex_deg",
	"THUMB_cmc_flex_deg",
	"THUMB_mcp_flex_deg",
	"THUMB_ip_flex_deg",
]

ANGLE_GROUPS: Dict[str, List[str]] = {
	"upper_limb": INTEREST_ANGLE_COLUMNS_UPPER_LIMB,
	"fingers": INTEREST_ANGLE_COLUMNS_FINGERS,
}

# Pre-specified anatomical families. Do not mix limb and finger angles in one
# PCA, and do not split the seven-dimensional limb family into extra analyses.
ACTIVE_ANGLE_GROUPS = list(ANGLE_GROUPS)

PERIODIC_ANGLE_COLUMNS = {
	"wrist_flex_ext_deg180",
	"wrist_dev_deg",
	"wrist_prosup_deg",
	"shoulder_plane_deg",
	"shoulder_axial_proxy_deg",
}

NORMALIZED_LENGTH = 200
PHASES = ["grasp"]
#PHASES = ["grasp"]
STAT_METHODS = ["mean_sem", "median_se"]
MAX_PCS_TO_PLOT = 8
MAX_PCS_TO_SAVE = 10

# Segmentation settings (from segmentation_test_four_stage_debug.py)
FS = 60.0
USE_LOWPASS = True
LOWPASS_CUTOFF_HZ = 8.0
LOWPASS_ORDER = 2
MARKER_SEG = "WRIST_y"
PROMINENCE_FACTOR = 0.6
MIN_DISTANCE_S = 0.3
THRESHOLD_LOW = -50
THRESHOLD_HIGH = 180
MIN_SEG_LEN_FRAMES = 2


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def compute_se_median(values: np.ndarray) -> float:
	n = len(values)
	if n < 2:
		return 0.0
	sd = np.std(values, ddof=1)
	return np.sqrt(np.pi / (2 * n)) * sd


def compute_statistics(values: np.ndarray, method: str = "mean_sem") -> Tuple[float, float, float]:
	vals = np.asarray(values, dtype=float)
	vals = vals[np.isfinite(vals)]
	n = len(vals)
	if n < 1:
		return np.nan, np.nan, np.nan

	if method == "mean_sem":
		center = float(np.mean(vals))
		if n < 2:
			return center, center, center
		err = float(np.std(vals, ddof=1) / np.sqrt(n))
		return center, center - err, center + err

	if method == "median_se":
		center = float(np.median(vals))
		err = float(compute_se_median(vals))
		return center, center - err, center + err

	raise ValueError(f"Unknown method: {method}")


def _interp_fill_1d(x: np.ndarray) -> np.ndarray:
	s = pd.Series(x, dtype=float).interpolate(limit_direction="both")
	return s.to_numpy(dtype=float)


def lowpass_safe(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
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


def make_segments_four_stage(y: np.ndarray, fs: float) -> List[Tuple[str, int, int]]:
	y = np.asarray(y, dtype=float)
	if np.isnan(y).any():
		y = _interp_fill_1d(y)

	y_f = lowpass_safe(y, LOWPASS_CUTOFF_HZ, fs, LOWPASS_ORDER) if USE_LOWPASS else y
	prom = max(np.nanstd(y_f) * PROMINENCE_FACTOR, 1e-6)
	distance = max(int(fs * MIN_DISTANCE_S), 1)

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

		crossing_low_up = None
		for j, val in enumerate(y_f[trough_start:peak_idx + 1]):
			if val >= THRESHOLD_LOW:
				crossing_low_up = trough_start + j
				break
		if crossing_low_up is None:
			continue

		crossing_high_up = None
		for j, val in enumerate(y_f[crossing_low_up:peak_idx + 1]):
			if val >= THRESHOLD_HIGH:
				crossing_high_up = crossing_low_up + j
				break
		if crossing_high_up is None:
			continue

		crossing_high_down = None
		for j, val in enumerate(y_f[peak_idx:trough_end + 1]):
			if val <= THRESHOLD_HIGH:
				crossing_high_down = peak_idx + j
				break
		if crossing_high_down is None:
			continue

		crossing_low_down = None
		for j, val in enumerate(y_f[crossing_high_down:trough_end + 1]):
			if val <= THRESHOLD_LOW:
				crossing_low_down = crossing_high_down + j
				break
		if crossing_low_down is None:
			continue

		grasp_start = prev_crossing_low_down if prev_crossing_low_down is not None else trough_start
		segs.append(("grasp", int(grasp_start), int(crossing_low_up)))
		segs.append(("carrying", int(crossing_low_up), int(crossing_high_up)))
		segs.append(("placement", int(crossing_high_up), int(peak_idx)))
		segs.append(("return", int(peak_idx), int(crossing_low_down)))

		prev_crossing_low_down = crossing_low_down

	# Remove first grasp in each block: first cycle has no grasp segment.
	first_grasp_removed = False
	segs_filtered: List[Tuple[str, int, int]] = []
	for phase, start, end in segs:
		if phase == "grasp" and not first_grasp_removed:
			first_grasp_removed = True
			continue
		segs_filtered.append((phase, start, end))

	return segs_filtered


def normalize_to_length(q_trial: np.ndarray, target_length: int = NORMALIZED_LENGTH) -> np.ndarray:
	t_orig, n_dof = q_trial.shape
	if t_orig < 2:
		return np.tile(q_trial[0], (target_length, 1))

	t0 = np.linspace(0.0, 1.0, t_orig)
	t1 = np.linspace(0.0, 1.0, target_length)
	out = np.zeros((target_length, n_dof), dtype=float)
	for i in range(n_dof):
		f = interp1d(t0, q_trial[:, i], kind="linear", bounds_error=False, fill_value="extrapolate")
		out[:, i] = f(t1)
	return out


def unwrap_deg_with_nans(values: np.ndarray) -> np.ndarray:
	"""Unwrap each contiguous finite run of a one-dimensional degree series."""
	values = np.asarray(values, dtype=float)
	out = values.copy()
	finite_idx = np.flatnonzero(np.isfinite(values))
	if finite_idx.size == 0:
		return out
	split_at = np.flatnonzero(np.diff(finite_idx) > 1) + 1
	for idx in np.split(finite_idx, split_at):
		out[idx] = np.rad2deg(np.unwrap(np.deg2rad(values[idx])))
	return out


def unwrap_periodic_segment(q_segment: np.ndarray, angle_columns: List[str]) -> np.ndarray:
	"""Make periodic coordinates continuous before linear time normalization."""
	out = np.asarray(q_segment, dtype=float).copy()
	for joint_idx, column in enumerate(angle_columns):
		if column in PERIODIC_ANGLE_COLUMNS:
			out[:, joint_idx] = unwrap_deg_with_nans(out[:, joint_idx])
	return out


def align_periodic_segment_branches(
	q_all: np.ndarray,
	angle_columns: List[str],
	branch_reference_deg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Align each segment to one 360-degree branch per periodic coordinate."""
	aligned = np.asarray(q_all, dtype=float).copy()
	n_dof = aligned.shape[2]
	if branch_reference_deg is None:
		reference = np.full(n_dof, np.nan, dtype=float)
		for joint_idx, column in enumerate(angle_columns[:n_dof]):
			if column not in PERIODIC_ANGLE_COLUMNS:
				continue
			values = aligned[:, :, joint_idx]
			values = values[np.isfinite(values)]
			if values.size:
				radians = np.deg2rad(values)
				reference[joint_idx] = np.rad2deg(
					np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
				)
	else:
		reference = np.asarray(branch_reference_deg, dtype=float).copy()
		if reference.shape != (n_dof,):
			raise ValueError("branch_reference_deg does not match the PCA variable count")

	for joint_idx, column in enumerate(angle_columns[:n_dof]):
		if column not in PERIODIC_ANGLE_COLUMNS or not np.isfinite(reference[joint_idx]):
			continue
		for segment_idx in range(aligned.shape[0]):
			curve = aligned[segment_idx, :, joint_idx]
			finite = np.isfinite(curve)
			if not finite.any():
				continue
			turns = np.rint((reference[joint_idx] - np.mean(curve[finite])) / 360.0)
			aligned[segment_idx, :, joint_idx] += 360.0 * turns
	return aligned, reference


def select_angle_columns(
	df_ang: pd.DataFrame,
	interest_angle_columns: Optional[List[str]],
) -> List[str]:
	if interest_angle_columns is None:
		ignore = {"frame", "smooth_enable", "smooth_method", "smooth_window"}
		return [c for c in df_ang.columns if c.endswith("_deg") and c not in ignore]

	# Fixed-dimensional PCA requires exactly the same variables in every block.
	# Silently dropping a missing angle would create incomparable PCA bases.
	missing = [c for c in interest_angle_columns if c not in df_ang.columns]
	if missing:
		return []
	return list(interest_angle_columns)


def resolve_task_name_for_subject(task: str, subject: str) -> str:
	"""Resolve subject-specific trial prefix used in data folders/files."""
	return task_name_candidates(task, subject)[0]


def trial_paths(subject: str, day: int, task: str, block: int) -> Tuple[Path, Path]:
	resolved = first_existing_trial_paths(
		((ABSREF_ROOT, "_with_table_absref.csv"), (ANGLE_ROOT, "_joint_angles.csv")),
		subject,
		day,
		task,
		block,
	)
	if resolved is not None:
		return resolved
	subj_day = f"{subject}_Day{day}"
	task_name = resolve_task_name_for_subject(task, subject)
	trial = f"{task_name}_{block}"
	abs_path = ABSREF_ROOT / subject / subj_day / trial / f"{trial}_with_table_absref.csv"
	ang_path = ANGLE_ROOT / subject / subj_day / trial / f"{trial}_joint_angles.csv"
	return abs_path, ang_path


def extract_normalized_segments(
	subject: str,
	day: int,
	task: str,
	block: int,
	interest_angle_columns: Optional[List[str]],
) -> Dict[str, List[np.ndarray]]:
	out: Dict[str, List[np.ndarray]] = {p: [] for p in PHASES}
	abs_path, ang_path = trial_paths(subject, day, task, block)

	if not abs_path.exists() or not ang_path.exists():
		return out

	try:
		df_abs = pd.read_csv(abs_path)
		df_ang = pd.read_csv(ang_path)
	except Exception:
		return out

	if MARKER_SEG not in df_abs.columns:
		return out

	angle_cols = select_angle_columns(df_ang, interest_angle_columns)
	if len(angle_cols) == 0:
		return out

	n = min(len(df_abs), len(df_ang))
	if n < 3:
		return out

	canonical_segments = make_common_task_segments(
		task,
		subject,
		day,
		block,
		df_abs.iloc[:n].reset_index(drop=True),
		fs=FS,
	)
	segs = [("grasp", start, end) for start, end in canonical_segments]
	if not segs:
		return out

	q_all = df_ang[angle_cols].to_numpy(dtype=float)[:n, :]
	for phase, s, e in segs:
		if phase not in out:
			continue
		s = max(0, s)
		e = min(n - 1, e)
		if e - s + 1 < MIN_SEG_LEN_FRAMES:
			continue
		q_seg = q_all[s:e + 1, :]
		if q_seg.shape[0] < 2:
			continue
		q_seg = unwrap_periodic_segment(q_seg, angle_cols)
		out[phase].append(normalize_to_length(q_seg, NORMALIZED_LENGTH))

	return out


def create_line_plot_pca_ratio_5days(
	df_pca_long: pd.DataFrame,
	stat_method: str,
	output_dir: Path,
	task: str,
	angle_group: str,
	pc_index: int,
	phase: str = "all",
) -> None:
	"""PCA EVR line plot by day with jittered participant points and error bars."""
	df = df_pca_long[(df_pca_long["task"] == task) & (df_pca_long["angle_group"] == angle_group)].copy()
	df = df[df["pc_index"] == pc_index]
	if phase != "all":
		df = df[df["phase"] == phase]
	if df.empty:
		return

	day_list = [1, 2, 3, 4, 5]
	data_by_day: Dict[int, Dict[str, List[float]]] = {d: {} for d in day_list}
	for _, row in df.iterrows():
		day = int(row["day"])
		subj = str(row["subject"])
		val = float(row["explained_variance_ratio"])
		data_by_day[day].setdefault(subj, []).append(val)

	subject_day_means: Dict[int, Dict[str, float]] = {d: {} for d in day_list}
	for day in day_list:
		for subj, vals in data_by_day[day].items():
			subject_day_means[day][subj] = float(np.mean(vals))

	pop_by_day: Dict[int, List[float]] = {d: [] for d in day_list}
	for day in day_list:
		for subj in subject_day_means[day]:
			pop_by_day[day].append(subject_day_means[day][subj])

	fig, ax = plt.subplots(figsize=(12, 9))
	rng = np.random.default_rng(123)
	for day in day_list:
		vals = pop_by_day[day]
		if len(vals) == 0:
			continue
		x_jitter = rng.normal(day, 0.06, len(vals))
		ax.scatter(x_jitter, vals, alpha=0.3, s=40, color="#1f77b4", edgecolors="none")

	stats_rows = []
	for day in day_list:
		vals = pop_by_day[day]
		if len(vals) == 0:
			continue
		center, lower, upper = compute_statistics(np.array(vals), stat_method)
		stats_rows.append({"x": day, "center": center, "lower": lower, "upper": upper})

	stats_df = pd.DataFrame(stats_rows)
	if not stats_df.empty:
		ax.plot(stats_df["x"], stats_df["center"], "-", color="#1f77b4", linewidth=2.5, alpha=0.8)
		ax.errorbar(
			stats_df["x"],
			stats_df["center"],
			yerr=[stats_df["center"] - stats_df["lower"], stats_df["upper"] - stats_df["center"]],
			fmt="o",
			color="#1f77b4",
			markersize=8,
			markerfacecolor="white",
			markeredgewidth=2,
			elinewidth=2,
			linewidth=0,
			capsize=5,
			capthick=2,
		)

	ax.set_xlabel("Day")
	ax.set_ylabel("Explained Variance Ratio")
	phase_label = phase if phase != "all" else "combined"
	ax.set_title(
		f"Raw PCA EVR across 5 Days - {task} (PC{pc_index}, {angle_group}, {phase_label}, {stat_method})"
	)
	ax.set_xticks(day_list)
	ax.set_xticklabels([f"Day {d}" for d in day_list])
	ax.grid(True, alpha=0.3, axis="y")
	ax.set_ylim(0.0, 1.0)

	output_file = output_dir / (
		f"line_plot_pca_evr_5days_{task}_{angle_group}_pc{pc_index}_{phase_label}_{stat_method}.png"
	)
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def create_line_plot_pca_cumulative_5days(
	df_pca_long: pd.DataFrame,
	stat_method: str,
	output_dir: Path,
	task: str,
	angle_group: str,
	target_pc: int = 8,
	phase: str = "all",
) -> None:
	"""Cumulative EVR line plot for PC1 through target_pc, saved per task."""
	df = df_pca_long[(df_pca_long["task"] == task) & (df_pca_long["angle_group"] == angle_group)].copy()
	df = df[df["pc_index"] == target_pc]
	if phase != "all":
		df = df[df["phase"] == phase]
	if df.empty:
		return

	day_list = [1, 2, 3, 4, 5]
	data_by_day: Dict[int, Dict[str, List[float]]] = {d: {} for d in day_list}
	for _, row in df.iterrows():
		day = int(row["day"])
		subj = str(row["subject"])
		val = float(row["cumulative_explained_variance_ratio"])
		data_by_day[day].setdefault(subj, []).append(val)

	pop_by_day: Dict[int, List[float]] = {d: [] for d in day_list}
	for day in day_list:
		for vals in data_by_day[day].values():
			pop_by_day[day].append(float(np.mean(vals)))

	fig, ax = plt.subplots(figsize=(12, 9))
	rng = np.random.default_rng(123)
	color = TASK_COLORS.get(task, "#1f77b4")
	for day in day_list:
		vals = pop_by_day[day]
		if len(vals) == 0:
			continue
		x_jitter = rng.normal(day, 0.06, len(vals))
		ax.scatter(x_jitter, vals, alpha=0.3, s=40, color=color, edgecolors="none")

	stats_rows = []
	for day in day_list:
		vals = pop_by_day[day]
		if len(vals) == 0:
			continue
		center, lower, upper = compute_statistics(np.array(vals), stat_method)
		stats_rows.append({"x": day, "center": center, "lower": lower, "upper": upper})

	stats_df = pd.DataFrame(stats_rows)
	if not stats_df.empty:
		ax.plot(stats_df["x"], stats_df["center"], "-", color=color, linewidth=2.5, alpha=0.85, label=task_label(task))
		ax.errorbar(
			stats_df["x"],
			stats_df["center"],
			yerr=[stats_df["center"] - stats_df["lower"], stats_df["upper"] - stats_df["center"]],
			fmt="o",
			color=color,
			markersize=8,
			markerfacecolor="white",
			markeredgewidth=2,
			elinewidth=2,
			linewidth=0,
			capsize=5,
			capthick=2,
		)

	phase_label = phase if phase != "all" else "combined"
	ax.set_xlabel("Day")
	ax.set_ylabel(f"PC1-PC{target_pc} Cumulative Explained Variance Ratio")
	ax.set_title(
		f"Raw PCA cumulative EVR across 5 Days - {task_label(task)} "
		f"(PC1-PC{target_pc}, {angle_group}, {phase_label}, {stat_method})"
	)
	ax.set_xticks(day_list)
	ax.set_xticklabels([f"Day {d}" for d in day_list])
	ax.grid(True, alpha=0.3, axis="y")
	ax.legend(loc="upper left")
	ax.set_ylim(0.0, 1.0)

	output_file = output_dir / (
		f"line_plot_pca_cumulative_evr_5days_{task}_{angle_group}_pc1to{target_pc}_{phase_label}_{stat_method}.png"
	)
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def create_line_plot_pca_ratio_5days_multitask(
	df_pca_long: pd.DataFrame,
	stat_method: str,
	output_dir: Path,
	angle_group: str,
	pc_index: int,
	phase: str = "all",
) -> None:
	"""Planned pairwise raw PCA EVR line plots by day."""
	df = df_pca_long[df_pca_long["angle_group"] == angle_group].copy()
	df = df[df["pc_index"] == pc_index]
	if phase != "all":
		df = df[df["phase"] == phase]
	if df.empty:
		return

	day_list = [1, 2, 3, 4, 5]
	phase_label = phase if phase != "all" else "combined"
	rng = np.random.default_rng(123)

	for pair in PLANNED_TASK_PAIRS:
		pair_df = df[df["task"].isin(pair)].copy()
		if not pair_has_both_tasks(pair_df, pair):
			continue

		fig, ax = plt.subplots(figsize=(12, 9))
		for task in pair:
			tdf = pair_df[pair_df["task"] == task]
			if tdf.empty:
				continue
			data_by_day: Dict[int, Dict[str, List[float]]] = {d: {} for d in day_list}
			for _, row in tdf.iterrows():
				day = int(row["day"])
				subj = str(row["subject"])
				val = float(row["explained_variance_ratio"])
				data_by_day[day].setdefault(subj, []).append(val)

			pop_by_day: Dict[int, List[float]] = {d: [] for d in day_list}
			for day in day_list:
				for vals in data_by_day[day].values():
					pop_by_day[day].append(float(np.mean(vals)))

			color = TASK_COLORS.get(task, "#1f77b4")
			for day in day_list:
				vals = pop_by_day[day]
				if len(vals) == 0:
					continue
				x_jitter = rng.normal(day, 0.06, len(vals))
				ax.scatter(x_jitter, vals, alpha=0.28, s=35, color=color, edgecolors="none")

			stats_rows = []
			for day in day_list:
				vals = pop_by_day[day]
				if len(vals) == 0:
					continue
				center, lower, upper = compute_statistics(np.array(vals), stat_method)
				stats_rows.append({"x": day, "center": center, "lower": lower, "upper": upper})

			stats_df = pd.DataFrame(stats_rows)
			if not stats_df.empty:
				ax.plot(stats_df["x"], stats_df["center"], "-", color=color, linewidth=2.5, alpha=0.85, label=task_label(task))
				ax.errorbar(
					stats_df["x"],
					stats_df["center"],
					yerr=[stats_df["center"] - stats_df["lower"], stats_df["upper"] - stats_df["center"]],
					fmt="o",
					color=color,
					markersize=7,
					markerfacecolor="white",
					markeredgewidth=2,
					elinewidth=1.8,
					linewidth=0,
					capsize=4,
					capthick=1.8,
				)

		pair_label = f"{task_label(pair[0])} vs {task_label(pair[1])}"
		ax.set_xlabel("Day")
		ax.set_ylabel("Explained Variance Ratio")
		ax.set_title(f"Raw PCA EVR across 5 Days - {pair_label} (PC{pc_index}, {angle_group}, {phase_label}, {stat_method})")
		ax.set_xticks(day_list)
		ax.set_xticklabels([f"Day {d}" for d in day_list])
		ax.grid(True, alpha=0.3, axis="y")
		ax.legend(loc="upper left")
		ax.set_ylim(0.0, 1.0)

		output_file = output_dir / f"line_plot_pca_evr_5days_pairwise_{comparison_file_label(pair)}_{angle_group}_pc{pc_index}_{phase_label}_{stat_method}.png"
		fig.tight_layout()
		fig.savefig(output_file, dpi=300, bbox_inches="tight")
		plt.close(fig)
def segments_to_matrix(
	q_segments: List[np.ndarray],
	angle_columns: List[str],
	branch_reference_deg: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
	"""Convert (N,T,D) segment list into a valid sample matrix (S,D)."""
	if len(q_segments) == 0:
		return None

	q_all = np.stack(q_segments, axis=0)  # (N, T, D)
	n_seg, n_time, n_dof = q_all.shape
	if n_dof < 1:
		return None
	q_all, branch_reference = align_periodic_segment_branches(
		q_all, angle_columns, branch_reference_deg
	)

	X = q_all.reshape(n_seg * n_time, n_dof)
	valid_mask = np.all(np.isfinite(X), axis=1)
	X = X[valid_mask]
	if X.shape[0] < 2:
		return None
	return X, branch_reference


def fit_fixed_pca_from_segments(
	q_segments: List[np.ndarray], angle_columns: List[str]
) -> Optional[Dict[str, np.ndarray]]:
	"""Fit fixed PCA basis from Day1 segments."""
	matrix_result = segments_to_matrix(q_segments, angle_columns)
	if matrix_result is None:
		return None
	X, branch_reference = matrix_result

	X_centered = X - np.mean(X, axis=0, keepdims=True)
	if not np.any(np.abs(X_centered) > 0):
		return None

	try:
		_, svals, vt = np.linalg.svd(X_centered, full_matrices=False)
	except np.linalg.LinAlgError:
		return None

	if len(svals) == 0:
		return None

	den = max(X_centered.shape[0] - 1, 1)
	variance = (svals ** 2) / den
	total_var = float(np.sum(variance))
	if total_var <= 0:
		return None

	ev_ratio_train = variance / total_var
	cum_ratio_train = np.cumsum(ev_ratio_train)

	return {
		"mean": np.mean(X, axis=0),
		"components": vt,
		"train_explained_variance": variance,
		"train_explained_variance_ratio": ev_ratio_train,
		"train_cumulative_explained_variance_ratio": cum_ratio_train,
		"n_train_samples": np.array([int(X_centered.shape[0])], dtype=int),
		"periodic_branch_reference_deg": branch_reference,
	}


def compute_fixed_basis_evr_for_segments(
	q_segments: List[np.ndarray],
	pca_model: Dict[str, np.ndarray],
	angle_columns: List[str],
) -> Optional[Dict[str, np.ndarray]]:
	"""Project segments onto fixed PCA basis and compute EVR in that basis."""
	matrix_result = segments_to_matrix(
		q_segments,
		angle_columns,
		pca_model.get("periodic_branch_reference_deg"),
	)
	if matrix_result is None:
		return None
	X, _ = matrix_result

	mean_vec = pca_model["mean"]
	components = pca_model["components"]
	if X.shape[1] != components.shape[1]:
		return None

	X_centered = X - mean_vec[None, :]
	scores = X_centered @ components.T
	if scores.shape[0] < 2:
		return None

	variance = np.var(scores, axis=0, ddof=1)
	total_var = float(np.sum(variance))
	if total_var <= 0:
		return None

	ev_ratio = variance / total_var
	cum_ratio = np.cumsum(ev_ratio)

	return {
		"explained_variance": variance,
		"explained_variance_ratio": ev_ratio,
		"cumulative_explained_variance_ratio": cum_ratio,
		"n_samples": np.array([int(X.shape[0])], dtype=int),
	}


def get_active_angle_groups_ordered() -> List[str]:
	"""Return valid angle groups in user-defined order (deduplicated)."""
	active_groups: List[str] = []
	seen = set()
	for g in ACTIVE_ANGLE_GROUPS:
		if not isinstance(g, str) or len(g) == 0:
			continue
		if g in seen:
			continue
		if g not in ANGLE_GROUPS:
			print(f"[skip] Unknown angle group: {g}")
			continue
		seen.add(g)
		active_groups.append(g)
	return active_groups


def main() -> None:
	ensure_dir(DATA_OUT_DIR)
	ensure_dir(VIS_OUT_DIR)
	task_vis_dirs: Dict[str, Path] = {task: VIS_OUT_DIR / task for task in TASK_LIST}
	for d in task_vis_dirs.values():
		ensure_dir(d)
	multitask_vis_dir = VIS_OUT_DIR / "multitask"
	ensure_dir(multitask_vis_dir)

	active_groups = get_active_angle_groups_ordered()
	if len(active_groups) == 0:
		print("[stop] No valid angle groups in ACTIVE_ANGLE_GROUPS.")
		return
	multitask_group_dirs: Dict[str, Path] = {
		angle_group: multitask_vis_dir / angle_group for angle_group in active_groups
	}
	for d in multitask_group_dirs.values():
		ensure_dir(d)

	overall_summary_rows: List[Dict[str, object]] = []
	overall_pca_long_rows: List[Dict[str, object]] = []
	saved_summary_files: List[Path] = []
	saved_pca_long_files: List[Path] = []

	for idx, angle_group in enumerate(active_groups, start=1):
		group_cols = ANGLE_GROUPS.get(angle_group)
		summary_rows: List[Dict[str, object]] = []
		pca_long_rows: List[Dict[str, object]] = []

		print(f"\n=== Angle group [{idx}/{len(active_groups)}]: {angle_group} ===")
		for task in TASK_LIST:
			print(f"\n=== Task: {task} ===")
			for subject in SUBJECT_LIST:
				# Fit one fixed PCA per subject-task-angle_group from Day1 all blocks/phases.
				day1_segments_all: List[np.ndarray] = []
				for block in BLOCK_LIST:
					phase_segments_day1 = extract_normalized_segments(subject, 1, task, block, group_cols)
					for phase in PHASES:
						day1_segments_all.extend(phase_segments_day1.get(phase, []))

				pca_model = fit_fixed_pca_from_segments(day1_segments_all, group_cols)
				if pca_model is None:
					print(f"[skip] {subject}/{task}/{angle_group}: cannot fit fixed Day1 PCA")
					continue

				for day in available_days_for_task(task):
					for block in BLOCK_LIST:
						phase_segments = extract_normalized_segments(subject, day, task, block, group_cols)

						for phase in PHASES:
							segs = phase_segments.get(phase, [])

							if len(segs) == 0:
								continue

							try:
								m = compute_fixed_basis_evr_for_segments(segs, pca_model, group_cols)
							except Exception:
								continue
							if m is None:
								continue

							evr = m["explained_variance_ratio"]
							cum = m["cumulative_explained_variance_ratio"]
							n_comp = int(len(evr))

							row: Dict[str, object] = {
								"angle_group": angle_group,
								"subject": subject,
								"task": task,
								"day": day,
								"block": block,
								"phase": phase,
								"n_segments": len(segs),
								"n_samples_for_pca": int(m["n_samples"][0]),
								"pca_train_day": 1,
								"pca_train_n_samples": int(pca_model["n_train_samples"][0]),
								"n_dof": int(np.stack(segs, axis=0).shape[2]),
								"pca_n_components": n_comp,
							}

							n_save = min(len(evr), MAX_PCS_TO_SAVE)
							for pc_idx, (evr_i, cum_i) in enumerate(zip(evr[:n_save], cum[:n_save]), start=1):
								row[f"PCA_EVR_PC{pc_idx}"] = float(evr_i)
								row[f"PCA_CUM_EVR_PC{pc_idx}"] = float(cum_i)
								pca_long_rows.append(
									{
										"angle_group": angle_group,
										"subject": subject,
										"task": task,
										"day": day,
										"block": block,
										"phase": phase,
										"n_segments": len(segs),
										"pc_index": pc_idx,
										"explained_variance_ratio": float(evr_i),
										"cumulative_explained_variance_ratio": float(cum_i),
									}
								)

							summary_rows.append(row)

							print(
								f"[OK] {subject} Day{day} {task}_{block} {phase} ({angle_group}): "
								f"n={len(segs)}, PC1={float(evr[0]):.6f}"
							)

		df_summary = pd.DataFrame(summary_rows)
		df_pca_long = pd.DataFrame(pca_long_rows)

		summary_csv = DATA_OUT_DIR / f"raw_pca_four_stage_summary_{angle_group}.csv"
		pca_long_csv = DATA_OUT_DIR / f"raw_pca_four_stage_long_{angle_group}.csv"
		df_summary.to_csv(summary_csv, index=False)
		df_pca_long.to_csv(pca_long_csv, index=False)
		saved_summary_files.append(summary_csv)
		saved_pca_long_files.append(pca_long_csv)

		overall_summary_rows.extend(summary_rows)
		overall_pca_long_rows.extend(pca_long_rows)

		# PCA EVR visualizations (5-day aggregated, task-wise + multitask) per group.
		if not df_pca_long.empty:
			max_pc_in_data = int(df_pca_long["pc_index"].max())
			pcs_to_plot = list(range(1, min(max_pc_in_data, MAX_PCS_TO_PLOT) + 1))
			plot_phases = PHASES
			# Visualizations are grasp-only; for Pegs this corresponds to right-to-left grasp.
			for stat_method in STAT_METHODS:
				for phase in plot_phases:
					for pc_index in pcs_to_plot:
						for task in TASK_LIST:
							create_line_plot_pca_ratio_5days(
								df_pca_long,
								stat_method,
								task_vis_dirs[task],
								task,
								angle_group,
								pc_index,
								phase=phase,
							)
						create_line_plot_pca_ratio_5days_multitask(
							df_pca_long,
							stat_method,
							multitask_group_dirs[angle_group],
							angle_group,
							pc_index,
							phase=phase,
						)
					if max_pc_in_data >= 8:
						for task in TASK_LIST:
							create_line_plot_pca_cumulative_5days(
								df_pca_long,
								stat_method,
								task_vis_dirs[task],
								task,
								angle_group,
								target_pc=8,
								phase=phase,
							)

	# Optional combined files when multiple groups are requested.
	if len(active_groups) > 1:
		group_tag = "__".join(active_groups)
		overall_summary_csv = DATA_OUT_DIR / f"raw_pca_four_stage_summary_{group_tag}.csv"
		overall_long_csv = DATA_OUT_DIR / f"raw_pca_four_stage_long_{group_tag}.csv"
		pd.DataFrame(overall_summary_rows).to_csv(overall_summary_csv, index=False)
		pd.DataFrame(overall_pca_long_rows).to_csv(overall_long_csv, index=False)

	print("\n" + "=" * 80)
	for p in saved_summary_files:
		print(f"Done. Summary saved: {p}")
	for p in saved_pca_long_files:
		print(f"Done. PCA long-format saved: {p}")
	print(f"Done. Visualizations saved to: {VIS_OUT_DIR}")
	print("=" * 80)


if __name__ == "__main__":
	main()


