#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raw-angle JDV analysis with canonical grasp segmentation.

Differences from H_NSTS/JDV_analysis.py:
- Uses raw joint angles directly (no UCM decomposition, no Jacobian projection).
- Uses D_kinematic/grasp_kinematic_common.py for every grasp window.

Reference baseline:
- For each (subject, task, phase), baseline is the mean normalized trajectory
  from Day5 Block5 of the same subject/task/phase.

JDV (raw) definition:
- For each segment trial (normalized to fixed length),
  delta(t, dof) = q_trial(t, dof) - q_baseline(t, dof)
- JDV_raw(t) = mean_trials( sum_dof(|delta|) / n_dof )
- JDV_raw_joint(t, dof) = mean_trials( |delta| )
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

try:
	import statsmodels.formula.api as smf
except Exception:
	smf = None


# =========================
# Configuration
# =========================
DATA_ROOT = Path(
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/NullSpace_Data"
)
ABSREF_ROOT = DATA_ROOT / "Checked_Absref_data"
ANGLE_ROOT = DATA_ROOT / "Joint_angles_time_series"
DATA_OUT_DIR = Path(
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/NullSpace_Data/raw_jdv_four_stage"
)
VIS_OUT_DIR = Path(
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis/Results_2/raw_jdv_four_stage"
)

SUBJECT_LIST = [
	"PT_06", "PT_07", "PT_08", "PT_09", "PT_10", "PT_11", "PT_12", "PT_13", "PT_14",
	"PT_15", "PT_16", "PT_17", "PT_18", "PT_19", "PT_20", "PT_21", "PT_22", "PT_23",
	"PT_24", "PT_25", "PT_26",
]

# Subjects where seated-task trials are stored under *_First naming.
SUBJECTS_WITH_FIRST_NAMING = {"PT_23", "PT_24", "PT_25", "PT_26"}

DAY_LIST = [1, 2, 3, 4, 5]
TASK_LIST = ["Tapes", "Tapes_Standing", "Pegs", "Pegs_Standing", "Coins", "Cognitive"]
TASK_AVAILABLE_DAYS = {"Cognitive": [1, 3, 5]}
TASK_DISPLAY_NAMES = {
	"Tapes": "Tapes seated",
	"Tapes_Seated": "Tapes seated",
	"Tapes_Standing": "Tapes standing",
	"Pegs": "Pegs seated",
	"Pegs_Seated": "Pegs seated",
	"Pegs_Standing": "Pegs standing",
	"Coins": "Coins",
	"Cognitive": "Cognitive",
}
TASK_PAIR_COMPARISONS = [
	("Tapes", "Coins"),
	("Tapes", "Tapes_Standing"),
	("Tapes", "Cognitive"),
	("Tapes", "Pegs"),
]


def task_display_name(task: str) -> str:
	return TASK_DISPLAY_NAMES.get(str(task).strip(), str(task).strip())


def task_file_label(task: str) -> str:
	return task_display_name(task).lower().replace(" ", "_")


def comparison_file_label(comparison_pair: Tuple[str, str]) -> str:
	return "_vs_".join(task_file_label(t) for t in comparison_pair)


def task_color(task: str) -> str:
	name = str(task).strip()
	if name in {"Tapes", "Tapes_Seated"}:
		return "#F0E442"
	if name == "Tapes_Standing":
		return "#009E73"
	if name in {"Pegs", "Pegs_Seated"}:
		return "#0072B2"
	if name == "Pegs_Standing":
		return "#17becf"
	if name == "Coins":
		return "#7F7F7F"
	if name == "Cognitive":
		return "#E69F00"
	return "#4a4a4a"



def available_days_for_task(task: str):
	return TASK_AVAILABLE_DAYS.get(task, DAY_LIST)
BLOCK_LIST = [1, 2, 3, 4, 5]

# Angle groups for quick switching. Set ACTIVE_ANGLE_GROUPS to run desired groups.
INTEREST_ANGLE_COLUMNS_UPPER_LIMB = [
	"wrist_flex_ext_deg180",
	"wrist_dev_deg",
	"wrist_prosup_deg",
	"elbow_flex_deg",
	"shoulder_elev_deg",
	"shoulder_plane_deg",
	"shoulder_axial_proxy_deg",
]

INTEREST_ANGLE_COLUMNS_UPPER_LIMB_WITHOUT_WRIST = [	"elbow_flex_deg",
	"shoulder_elev_deg",
	"shoulder_plane_deg",
	"shoulder_axial_proxy_deg",
]


INTEREST_ANGLE_COLUMNS_WRIST = [	
    "wrist_flex_ext_deg180",
	"wrist_dev_deg",
	"wrist_prosup_deg"]


INTEREST_ANGLE_COLUMNS_WRIST_flex_ext = [	
    "wrist_flex_ext_deg180"]

INTEREST_ANGLE_COLUMNS_INDEX = [
	"INDEX_mcp_flex_deg",
	"INDEX_pip_flex_deg",
	"INDEX_dip_flex_deg",
]

INTEREST_ANGLE_COLUMNS_MIDDLE = [
	"MIDDLE_mcp_flex_deg",
	"MIDDLE_pip_flex_deg",
	"MIDDLE_dip_flex_deg",
]

INTEREST_ANGLE_COLUMNS_RING = [
	"RING_mcp_flex_deg",
	"RING_pip_flex_deg",
	"RING_dip_flex_deg",
]

INTEREST_ANGLE_COLUMNS_PINKY = [
	"PINKY_mcp_flex_deg",
	"PINKY_pip_flex_deg",
	"PINKY_dip_flex_deg",
]

INTEREST_ANGLE_COLUMNS_THUMB = [
	"THUMB_cmc_flex_deg",
	"THUMB_mcp_flex_deg",
	"THUMB_ip_flex_deg",
]

INTEREST_ANGLE_COLUMNS_THIRD_THUMB = [
	"THIRD_THUMB_mcp_flex_deg",
	"THIRD_THUMB_pip_flex_deg",
	"THIRD_THUMB_dip_flex_deg",
]

INTEREST_ANGLE_COLUMNS_FINGERS = (
	INTEREST_ANGLE_COLUMNS_INDEX
	+ INTEREST_ANGLE_COLUMNS_MIDDLE
	+ INTEREST_ANGLE_COLUMNS_RING
	+ INTEREST_ANGLE_COLUMNS_PINKY
	+ INTEREST_ANGLE_COLUMNS_THUMB
)

INTEREST_ANGLE_COLUMNS_ALL = INTEREST_ANGLE_COLUMNS_UPPER_LIMB + INTEREST_ANGLE_COLUMNS_FINGERS

ANGLE_GROUPS: Dict[str, Optional[List[str]]] = {
	"all": INTEREST_ANGLE_COLUMNS_ALL,
	"upper_limb": INTEREST_ANGLE_COLUMNS_UPPER_LIMB,
	"upper_limb_without_wrist": INTEREST_ANGLE_COLUMNS_UPPER_LIMB_WITHOUT_WRIST,
	"fingers": INTEREST_ANGLE_COLUMNS_FINGERS,
	"index": INTEREST_ANGLE_COLUMNS_INDEX,
	"middle": INTEREST_ANGLE_COLUMNS_MIDDLE,
	"ring": INTEREST_ANGLE_COLUMNS_RING,
	"pinky": INTEREST_ANGLE_COLUMNS_PINKY,
	"thumb": INTEREST_ANGLE_COLUMNS_THUMB,
	"third_thumb": INTEREST_ANGLE_COLUMNS_THIRD_THUMB,
	"wrist": INTEREST_ANGLE_COLUMNS_WRIST,
	"wrist_flex_ext": INTEREST_ANGLE_COLUMNS_WRIST_flex_ext,
	"all_deg_in_file": None,
}

# Choose one or more groups to run.
#ACTIVE_ANGLE_GROUPS = ["fingers"]
#ACTIVE_ANGLE_GROUPS = ["upper_limb"]
ACTIVE_ANGLE_GROUPS = ["wrist_flex_ext","upper_limb_without_wrist", "wrist"]

# Columns generated from atan2-style signed angles.  Keep the existing display
# names and output schema, but treat these coordinates as 360-degree periodic
# whenever trajectories are interpolated, averaged, or differenced.
PERIODIC_ANGLE_COLUMNS = {
	"wrist_flex_ext_deg180",
	"wrist_dev_deg",
	"wrist_prosup_deg",
	"shoulder_plane_deg",
	"shoulder_axial_proxy_deg",
}

NORMALIZED_LENGTH = 200
PHASES = ["grasp"]
STAT_METHODS = ["mean_sem", "median_se"]

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


def mean_angle_trajectory(q_segments: List[np.ndarray], angle_columns: Optional[List[str]]) -> np.ndarray:
	"""Average trajectories, using a circular mean for periodic coordinates."""
	q_all = np.stack(q_segments, axis=0)
	mean_trajectory = np.mean(q_all, axis=0)
	if angle_columns is None:
		return mean_trajectory

	for joint_idx, column in enumerate(angle_columns):
		if joint_idx >= q_all.shape[2]:
			break
		if column not in PERIODIC_ANGLE_COLUMNS:
			continue
		radians = np.deg2rad(q_all[:, :, joint_idx])
		mean_trajectory[:, joint_idx] = np.rad2deg(
			np.arctan2(np.mean(np.sin(radians), axis=0), np.mean(np.cos(radians), axis=0))
		)
	return mean_trajectory


def shortest_angle_delta_deg(delta: np.ndarray, angle_columns: Optional[List[str]]) -> np.ndarray:
	"""Wrap periodic residuals to [-180, 180] while leaving linear joints unchanged."""
	out = np.asarray(delta, dtype=float).copy()
	if angle_columns is None:
		return out
	for joint_idx, column in enumerate(angle_columns):
		if joint_idx >= out.shape[-1]:
			break
		if column in PERIODIC_ANGLE_COLUMNS:
			radians = np.deg2rad(out[..., joint_idx])
			out[..., joint_idx] = np.rad2deg(np.arctan2(np.sin(radians), np.cos(radians)))
	return out


def select_angle_columns(
	df_ang: pd.DataFrame,
	interest_angle_columns: Optional[List[str]],
) -> List[str]:
	if interest_angle_columns is None:
		ignore = {"frame", "smooth_enable", "smooth_method", "smooth_window"}
		return [c for c in df_ang.columns if c.endswith("_deg") and c not in ignore]

	if any(c not in df_ang.columns for c in interest_angle_columns):
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


def _collect_phase_segments_for_trial(
	subject: str,
	day: int,
	block: int,
	task: str,
	phase: str,
	interest_angle_columns: Optional[List[str]],
) -> List[np.ndarray]:
	phase_segments = extract_normalized_segments(subject, day, task, block, interest_angle_columns)
	return phase_segments.get(phase, [])


def compute_baseline(
	subject: str,
	task: str,
	phase: str,
	interest_angle_columns: Optional[List[str]],
) -> Optional[np.ndarray]:
	"""
	Baseline selection rule:
	1) Prefer last day last two blocks: Day5-Block4 and Day5-Block5.
	2) If no valid segments there, fallback to last two valid blocks across all days.
	"""

	# Preferred baseline blocks on last day.
	preferred_blocks = [(5, 4), (5, 5)]
	baseline_segments: List[np.ndarray] = []
	for day, block in preferred_blocks:
		baseline_segments.extend(
			_collect_phase_segments_for_trial(subject, day, block, task, phase, interest_angle_columns)
		)

	if len(baseline_segments) > 0:
		return mean_angle_trajectory(baseline_segments, interest_angle_columns)

	# Fallback: find the last two available blocks across all day/block combinations.
	available: List[Tuple[int, int, List[np.ndarray]]] = []
	for day in sorted(DAY_LIST):
		for block in sorted(BLOCK_LIST):
			segs = _collect_phase_segments_for_trial(subject, day, block, task, phase, interest_angle_columns)
			if len(segs) > 0:
				available.append((day, block, segs))

	if len(available) == 0:
		return None

	last_two = available[-2:] if len(available) >= 2 else available[-1:]
	fallback_segments: List[np.ndarray] = []
	for _, _, segs in last_two:
		fallback_segments.extend(segs)

	if len(fallback_segments) == 0:
		return None

	return mean_angle_trajectory(fallback_segments, interest_angle_columns)


def create_line_plot_jdv_raw_5days(
	df_summary: pd.DataFrame,
	stat_method: str,
	output_dir: Path,
	task: str,
	angle_group: str,
	phase: str = "all",
) -> None:
	"""JDV raw line plot by day with jittered participant points and error bars."""
	df = df_summary[(df_summary["task"] == task) & (df_summary["angle_group"] == angle_group)].copy()
	if phase != "all":
		df = df[df["phase"] == phase]
	if df.empty:
		return

	day_list = [1, 2, 3, 4, 5]
	data_by_day: Dict[int, Dict[str, List[float]]] = {d: {} for d in day_list}
	for _, row in df.iterrows():
		day = int(row["day"])
		subj = str(row["subject"])
		val = float(row["JDV_raw_mean"])
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
	color = task_color(task)
	for day in day_list:
		vals = pop_by_day[day]
		if len(vals) == 0:
			continue
		x_jitter = rng.normal(day, 0.06, len(vals))
		ax.scatter(
			x_jitter,
			vals,
			alpha=0.85,
			s=90,
			color=color,
			edgecolors=color,
			linewidths=1.0,
		)

	stats_rows = []
	for day in day_list:
		vals = pop_by_day[day]
		if len(vals) == 0:
			continue
		center, lower, upper = compute_statistics(np.array(vals), stat_method)
		stats_rows.append({"x": day, "center": center, "lower": lower, "upper": upper})

	stats_df = pd.DataFrame(stats_rows)
	if not stats_df.empty:
		ax.plot(stats_df["x"], stats_df["center"], "-", color=color, linewidth=4.2, alpha=1.0)
		ax.errorbar(
			stats_df["x"],
			stats_df["center"],
			yerr=[stats_df["center"] - stats_df["lower"], stats_df["upper"] - stats_df["center"]],
			fmt="o",
			color=color,
			markersize=12,
			markerfacecolor="white",
			markeredgewidth=3.8,
			elinewidth=4.2,
			linewidth=0,
			capsize=7,
			capthick=4.2,
		)

	phase_label = phase if phase != "all" else "combined"
	ax.set_title("")
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.tick_params(axis="both", length=0)
	ax.grid(False)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_linewidth(3)
	ax.spines["bottom"].set_linewidth(3)
	ax.set_ylim(0, 30)

	output_file = output_dir / f"line_plot_jdv_raw_5days_{task}_{angle_group}_{phase_label}_{stat_method}.png"
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def create_line_plot_jdv_raw_5days_multitask(
	df_summary: pd.DataFrame,
	stat_method: str,
	output_dir: Path,
	angle_group: str,
	phase: str = "all",
	comparison_pair: Tuple[str, str] = ("Tapes", "Coins"),
) -> None:
	"""Raw JDV line plot for one pre-planned task comparison."""
	df = df_summary[df_summary["angle_group"] == angle_group].copy()
	if phase != "all":
		df = df[df["phase"] == phase]
	df = df[df["task"].isin(comparison_pair)]
	if df.empty:
		return

	day_list = [1, 2, 3, 4, 5]
	task_ordered = [task for task in comparison_pair if task in set(df["task"].dropna())]
	if len(task_ordered) == 0:
		return

	fig, ax = plt.subplots(figsize=(12, 9))
	rng = np.random.default_rng(123)

	for task in task_ordered:
		tdf = df[df["task"] == task]
		data_by_day: Dict[int, Dict[str, List[float]]] = {d: {} for d in day_list}
		for _, row in tdf.iterrows():
			day = int(row["day"])
			subj = str(row["subject"])
			val = float(row["JDV_raw_mean"])
			data_by_day[day].setdefault(subj, []).append(val)

		pop_by_day: Dict[int, List[float]] = {d: [] for d in day_list}
		for day in day_list:
			for subj, vals in data_by_day[day].items():
				pop_by_day[day].append(float(np.mean(vals)))

		color = task_color(task)
		for day in day_list:
			vals = pop_by_day[day]
			if len(vals) == 0:
				continue
			x_jitter = rng.normal(day, 0.06, len(vals))
			ax.scatter(
				x_jitter,
				vals,
				alpha=0.85,
				s=90,
				color=color,
				edgecolors=color,
				linewidths=2.0,
			)

		stats_rows = []
		for day in day_list:
			vals = pop_by_day[day]
			if len(vals) == 0:
				continue
			center, lower, upper = compute_statistics(np.array(vals), stat_method)
			stats_rows.append({"x": day, "center": center, "lower": lower, "upper": upper})

		stats_df = pd.DataFrame(stats_rows)
		if not stats_df.empty:
			ax.plot(stats_df["x"], stats_df["center"], "-", color=color, linewidth=6.0, alpha=1.0, label=task_display_name(task))
			ax.errorbar(
				stats_df["x"],
				stats_df["center"],
				yerr=[stats_df["center"] - stats_df["lower"], stats_df["upper"] - stats_df["center"]],
				fmt="o",
				color=color,
				markersize=12,
				markerfacecolor="white",
				markeredgewidth=3.8,
				elinewidth=4.2,
				linewidth=0,
				capsize=7,
				capthick=4.2,
			)

	phase_label = phase if phase != "all" else "combined"
	comparison_label = comparison_file_label(comparison_pair)
	ax.set_title("")
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.tick_params(axis="both", length=0)
	ax.grid(False)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_linewidth(3)
	ax.spines["bottom"].set_linewidth(3)
	ax.set_ylim(0, 30)
	legend = ax.legend(
		loc="upper left",
		markerscale=2.0,
		labelspacing=3,
		handlelength=2.8,
		handletextpad=0.0,
		borderpad=0.35,
		frameon=False,
	)
	if legend is not None:
		for text in legend.get_texts():
			text.set_text("")

	output_file = output_dir / f"line_plot_jdv_raw_5days_{comparison_label}_{angle_group}_{phase_label}_{stat_method}.png"
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def create_line_plot_jdv_raw_5days_multitask_annotated(
	df_summary: pd.DataFrame,
	stat_method: str,
	output_dir: Path,
	angle_group: str,
	phase: str = "all",
	comparison_pair: Tuple[str, str] = ("Tapes", "Coins"),
) -> None:
	"""Annotated raw JDV line plot for one pre-planned task comparison."""
	df = df_summary[df_summary["angle_group"] == angle_group].copy()
	if phase != "all":
		df = df[df["phase"] == phase]
	df = df[df["task"].isin(comparison_pair)]
	if df.empty:
		return

	day_list = [1, 2, 3, 4, 5]
	task_ordered = [task for task in comparison_pair if task in set(df["task"].dropna())]
	if len(task_ordered) == 0:
		return

	fig, ax = plt.subplots(figsize=(13, 9))
	rng = np.random.default_rng(123)

	y_global_max = 0.0
	stats_by_task: Dict[str, pd.DataFrame] = {}
	subject_counts_by_task_day: Dict[str, Dict[int, int]] = {}

	for task in task_ordered:
		tdf = df[df["task"] == task]
		data_by_day: Dict[int, Dict[str, List[float]]] = {d: {} for d in day_list}
		for _, row in tdf.iterrows():
			day = int(row["day"])
			subj = str(row["subject"])
			val = float(row["JDV_raw_mean"])
			data_by_day[day].setdefault(subj, []).append(val)

		pop_by_day: Dict[int, List[float]] = {d: [] for d in day_list}
		subject_counts_by_task_day[task] = {}
		for day in day_list:
			for subj, vals in data_by_day[day].items():
				pop_by_day[day].append(float(np.mean(vals)))
			subject_counts_by_task_day[task][day] = len(pop_by_day[day])

		color = task_color(task)
		for day in day_list:
			vals = pop_by_day[day]
			if len(vals) == 0:
				continue
			x_jitter = rng.normal(day, 0.055, len(vals))
			ax.scatter(
				x_jitter,
				vals,
				alpha=0.35,
				s=65,
				color=color,
				edgecolors="none",
			)
			y_global_max = max(y_global_max, float(np.max(vals)))

		stats_rows = []
		for day in day_list:
			vals = pop_by_day[day]
			if len(vals) == 0:
				continue
			center, lower, upper = compute_statistics(np.array(vals), stat_method)
			stats_rows.append({"x": day, "center": center, "lower": lower, "upper": upper})

		stats_df = pd.DataFrame(stats_rows)
		stats_by_task[task] = stats_df
		if not stats_df.empty:
			ax.plot(
				stats_df["x"],
				stats_df["center"],
				"-",
				color=color,
				linewidth=3.8,
				alpha=1.0,
				label=task_display_name(task),
			)
			ax.fill_between(
				stats_df["x"],
				stats_df["lower"],
				stats_df["upper"],
				color=color,
				alpha=0.14,
				linewidth=0,
			)
			ax.errorbar(
				stats_df["x"],
				stats_df["center"],
				yerr=[stats_df["center"] - stats_df["lower"], stats_df["upper"] - stats_df["center"]],
				fmt="o",
				color=color,
				markersize=9,
				markerfacecolor="white",
				markeredgewidth=2.0,
				elinewidth=2.0,
				linewidth=0,
				capsize=4,
				capthick=2.0,
			)
			y_global_max = max(y_global_max, float(stats_df["upper"].max()))

	phase_label = phase if phase != "all" else "combined"
	comparison_title = " vs ".join(task_display_name(t) for t in comparison_pair)
	comparison_label = comparison_file_label(comparison_pair)
	ax.set_title(
		f"Raw JDV across 5 days: {comparison_title} ({angle_group}, {phase_label})",
		fontsize=20,
		fontweight="bold",
		pad=14,
	)
	ax.set_xlabel("Day", fontsize=15)
	ax.set_ylabel("JDV (raw, deg)", fontsize=15)
	ax.set_xticks(day_list)
	ax.set_xticklabels([f"Day {d}" for d in day_list], fontsize=12)
	ax.tick_params(axis="y", labelsize=12)
	ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_linewidth(1.8)
	ax.spines["bottom"].set_linewidth(1.8)

	y_lim_top = max(30.0, y_global_max * 1.28 if y_global_max > 0 else 30.0)
	ax.set_ylim(0, y_lim_top)

	for task in task_ordered:
		stats_df = stats_by_task.get(task)
		if stats_df is None or stats_df.empty:
			continue
		color = task_color(task)
		for _, row in stats_df.iterrows():
			day = int(row["x"])
			center = float(row["center"])
			upper = float(row["upper"])
			n_subjects = subject_counts_by_task_day.get(task, {}).get(day, 0)
			label_text = f"{center:.2f}\n(n={n_subjects})"
			ax.annotate(
				label_text,
				xy=(day, upper),
				xytext=(0, 8),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=9,
				color=color,
			)

	legend = ax.legend(loc="upper left", frameon=False, fontsize=12, title="Task")
	if legend is not None and legend.get_title() is not None:
		legend.get_title().set_fontsize(12)

	method_text = "Mean ± SEM" if stat_method == "mean_sem" else "Median ± SE(median)"
	ax.text(
		0.99,
		0.015,
		f"Summary: {method_text}",
		transform=ax.transAxes,
		ha="right",
		va="bottom",
		fontsize=10,
		color="#404040",
	)

	output_file = output_dir / (
		f"line_plot_jdv_raw_5days_{comparison_label}_annotated_{angle_group}_{phase_label}_{stat_method}.png"
	)
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)

def aggregate_subject_day_task(df_summary: pd.DataFrame, angle_group: str, phase: str, comparison_pair: Tuple[str, str]) -> pd.DataFrame:
	df = df_summary[df_summary["angle_group"] == angle_group].copy()
	if phase != "all":
		df = df[df["phase"] == phase]
	df = df[df["task"].isin(comparison_pair)]
	if df.empty:
		return df
	return df.groupby(["subject", "task", "day", "angle_group", "phase"], as_index=False)["JDV_raw_mean"].mean()


def run_pairwise_task_statistics(
	df_summary: pd.DataFrame,
	output_dir: Path,
	angle_group: str,
	phase: str,
	comparison_pair: Tuple[str, str],
) -> pd.DataFrame:
	"""Save the planned Task x Day LMM for one task/reference pair."""
	df_pair = aggregate_subject_day_task(df_summary, angle_group, phase, comparison_pair)
	if df_pair.empty:
		return pd.DataFrame()

	phase_label = phase if phase != "all" else "combined"
	comparison_label = comparison_file_label(comparison_pair)
	lmm_rows: List[Dict[str, object]] = []
	if smf is None:
		lmm_rows.append({"angle_group": angle_group, "phase": phase_label, "comparison": comparison_label, "model": "JDV_raw_mean ~ task * day + (1|subject)", "status": "statsmodels unavailable"})
	else:
		try:
			lmm_df = df_pair.copy()
			comparison_task, reference_task = comparison_pair
			lmm_df["task_code"] = (lmm_df["task"] == comparison_task).astype(float)
			lmm_df["day_c"] = lmm_df["day"].astype(float) - 1.0
			model = smf.mixedlm(
				"JDV_raw_mean ~ task_code * day_c",
				lmm_df,
				groups=lmm_df["subject"],
			)
			result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
			conf = result.conf_int()
			for term in result.params.index:
				lmm_rows.append(
					{
						"angle_group": angle_group,
						"phase": phase_label,
						"comparison": comparison_label,
						"comparison_task": comparison_task,
						"reference_task": reference_task,
						"model": "JDV_raw_mean ~ task * day + (1|subject)",
						"status": "ok",
						"term": term,
						"estimate": float(result.params[term]),
						"std_error": float(result.bse.get(term, np.nan)),
						"z_value": float(result.tvalues.get(term, np.nan)),
						"p_value": float(result.pvalues.get(term, np.nan)),
						"ci_lower": float(conf.loc[term, 0]) if term in conf.index else np.nan,
						"ci_upper": float(conf.loc[term, 1]) if term in conf.index else np.nan,
					}
				)
		except Exception as exc:
			lmm_rows.append({"angle_group": angle_group, "phase": phase_label, "comparison": comparison_label, "model": "JDV_raw_mean ~ task * day + (1|subject)", "status": f"failed: {exc}"})

	lmm_file = output_dir / f"lmm_jdv_raw_5days_{comparison_label}_{angle_group}_{phase_label}.csv"
	lmm_table = pd.DataFrame(lmm_rows)
	lmm_table.to_csv(lmm_file, index=False)
	print(f"[save] LMM results saved: {lmm_file}")
	return lmm_table
def compute_jdv_against_baseline(
	q_segments: List[np.ndarray],
	baseline: np.ndarray,
	angle_columns: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
	"""Compute JDV raw time-series and joint-wise values."""
	q_all = np.stack(q_segments, axis=0)  # (N, T, D)
	delta = shortest_angle_delta_deg(q_all - baseline[None, :, :], angle_columns)
	n_dof = q_all.shape[2]

	jdv_raw_t = np.mean(np.sum(np.abs(delta), axis=2) / max(n_dof, 1), axis=0)  # (T,)
	jdv_joint_t = np.mean(np.abs(delta), axis=0)  # (T, D)

	return {
		"JDV_raw_t": jdv_raw_t,
		"JDV_raw_mean": np.array([float(np.mean(jdv_raw_t))]),
		"JDV_raw_joint_mean": jdv_joint_t.mean(axis=0),
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


def get_processed_data_file(
	output_dir: Path,
	angle_group: str,
	phase: str,
	stat_method: str,
) -> Path:
	phase_label = phase if phase != "all" else "combined"
	return output_dir / f"processed_jdv_raw_5days_multitask_{angle_group}_{phase_label}_{stat_method}.csv"


def has_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
	return all(col in df.columns for col in required_columns)


def main() -> None:
	ensure_dir(VIS_OUT_DIR)

	phase = "grasp"
	stat_method = "mean_sem"
	active_groups = get_active_angle_groups_ordered()
	if len(active_groups) == 0:
		print("[stop] No valid ACTIVE_ANGLE_GROUPS found; nothing to run.")
		return

	required_columns = [
		"angle_group",
		"subject",
		"task",
		"day",
		"block",
		"phase",
		"n_segments",
		"JDV_raw_mean",
	]

	for angle_group in active_groups:
		group_cols = ANGLE_GROUPS.get(angle_group)
		processed_data_file = get_processed_data_file(VIS_OUT_DIR, angle_group, phase, stat_method)
		df_summary: Optional[pd.DataFrame] = None

		if processed_data_file.exists():
			try:
				df_loaded = pd.read_csv(processed_data_file)
				if has_required_columns(df_loaded, required_columns) and not df_loaded.empty:
					df_summary = df_loaded
					print(f"[load] Processed data loaded: {processed_data_file}")
				else:
					print(f"[recompute] Cached file invalid or empty: {processed_data_file}")
			except Exception:
				print(f"[recompute] Failed to load cached file: {processed_data_file}")

		if df_summary is None:
			summary_rows: List[Dict[str, object]] = []

			print(f"\n=== Target output: line_plot_jdv_raw_5days_multitask_{angle_group}_{phase}_{stat_method}.png ===")
			for task in TASK_LIST:
				print(f"\n=== Task: {task} ===")
				for subject in SUBJECT_LIST:
					baseline = compute_baseline(subject, task, phase, group_cols)
					if baseline is None:
						continue

					for day in available_days_for_task(task):
						for block in BLOCK_LIST:
							phase_segments = extract_normalized_segments(subject, day, task, block, group_cols)
							segs = phase_segments.get(phase, [])

							if len(segs) == 0:
								continue

							try:
								m = compute_jdv_against_baseline(segs, baseline, group_cols)
							except Exception:
								continue

							summary_rows.append(
								{
									"angle_group": angle_group,
									"subject": subject,
									"task": task,
									"day": day,
									"block": block,
									"phase": phase,
									"n_segments": len(segs),
									"JDV_raw_mean": float(m["JDV_raw_mean"][0]),
								}
							)

							print(
								f"[OK] {subject} Day{day} {task}_{block} {phase} ({angle_group}): "
								f"n={len(segs)}, JDV={float(m['JDV_raw_mean'][0]):.6f}"
							)

			df_summary = pd.DataFrame(summary_rows)
			if df_summary.empty:
				print(f"[stop] No valid data found for {angle_group}/{phase}; plot not generated.")
				continue

			df_summary.to_csv(processed_data_file, index=False)
			print(f"[save] Processed data saved: {processed_data_file}")

		for task in TASK_LIST:
			task_output_dir = VIS_OUT_DIR / "per_task" / task
			ensure_dir(task_output_dir)
			create_line_plot_jdv_raw_5days(
				df_summary=df_summary,
				stat_method=stat_method,
				output_dir=task_output_dir,
				task=task,
				angle_group=angle_group,
				phase=phase,
			)

		pairwise_lmm_tables: List[pd.DataFrame] = []
		for comparison_pair in TASK_PAIR_COMPARISONS:
			comparison_label = comparison_file_label(comparison_pair)
			create_line_plot_jdv_raw_5days_multitask(
				df_summary=df_summary,
				stat_method=stat_method,
				output_dir=VIS_OUT_DIR,
				angle_group=angle_group,
				phase=phase,
				comparison_pair=comparison_pair,
			)

			create_line_plot_jdv_raw_5days_multitask_annotated(
				df_summary=df_summary,
				stat_method=stat_method,
				output_dir=VIS_OUT_DIR,
				angle_group=angle_group,
				phase=phase,
				comparison_pair=comparison_pair,
			)

			lmm_table = run_pairwise_task_statistics(
				df_summary=df_summary,
				output_dir=VIS_OUT_DIR,
				angle_group=angle_group,
				phase=phase,
				comparison_pair=comparison_pair,
			)
			if not lmm_table.empty:
				pairwise_lmm_tables.append(lmm_table)

			out_file = VIS_OUT_DIR / f"line_plot_jdv_raw_5days_{comparison_label}_{angle_group}_{phase}_{stat_method}.png"
			out_file_annotated = VIS_OUT_DIR / (
				f"line_plot_jdv_raw_5days_{comparison_label}_annotated_{angle_group}_{phase}_{stat_method}.png"
			)
			print("\n" + "=" * 80)
			print(f"Done. Figure saved: {out_file}")
			print(f"Done. Figure saved: {out_file_annotated}")
			print("=" * 80)

		if pairwise_lmm_tables:
			combined_lmm = pd.concat(pairwise_lmm_tables, ignore_index=True)
			combined_lmm["p_fdr_bh"] = np.nan
			term_values = combined_lmm.get(
				"term", pd.Series(index=combined_lmm.index, dtype=object)
			)
			p_values_all = pd.to_numeric(
				combined_lmm.get(
					"p_value", pd.Series(index=combined_lmm.index, dtype=float)
				),
				errors="coerce",
			)
			mask = (term_values == "task_code:day_c") & p_values_all.notna()
			indices = combined_lmm.index[mask]
			if len(indices):
				p_values = combined_lmm.loc[indices, "p_value"].to_numpy(float)
				order = np.argsort(p_values)
				ranked = p_values[order]
				adjusted = np.minimum.accumulate(
					(ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
				)[::-1]
				restored = np.empty_like(adjusted)
				restored[order] = np.clip(adjusted, 0.0, 1.0)
				combined_lmm.loc[indices, "p_fdr_bh"] = restored
			combined_lmm.to_csv(
				VIS_OUT_DIR / f"lmm_jdv_raw_all_planned_comparisons_{angle_group}_{phase}.csv",
				index=False,
			)


if __name__ == "__main__":
	main()




