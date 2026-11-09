#!/usr/bin/env python3
"""Minimal mathematical corrections to the legacy trajectory-relative UCM.

This file deliberately preserves the 2026-09-01 choices that define the
scientific estimand: seven elemental angles, repetition-specific ``(s, e_n, z)``
paths, robust repetition QC, temporal windows, raw DeltaV, and the legacy
figures.  Segmentation is delegated to the single active
``D_kinematic/grasp_kinematic_common.py`` implementation.  Its default policy
reproduces the 2026-09-01 thresholds and boundary handling for non-Pegs tasks,
while retaining the later corrected Pegs right-to-left direction rule.

The UCM calculation changes only the following mathematical points:

1. Interval Jacobians are aligned to interval midpoints before they are paired
   with position trajectories on normalised time.
2. Jacobians expressed in different repetition-specific task frames are not
   averaged element by element.  Each Jacobian is first converted to its exact
   joint-space task projector; a coordinate-invariant consensus projector is
   then obtained from the leading eigenspace of the mean projector.
3. Task/null projections are exact, mutually orthogonal SVD/eigen projectors.
   Total variance, the classical normalised DeltaV, and partition diagnostics
   are exported in addition to the legacy-compatible raw metrics.

The source file ``UCM_angle_process_more_robust_tt_tip_aug.py`` is not imported
or modified.  Outputs are isolated under
``UCM_more_robust_minimal_mathfix_common_inclusive``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks


KINEMATIC_COMMON_DIR = Path(__file__).resolve().parents[1] / "D_kinematic"
if str(KINEMATIC_COMMON_DIR) not in sys.path:
	sys.path.insert(0, str(KINEMATIC_COMMON_DIR))

from grasp_kinematic_common import (  # noqa: E402
	DEFAULT_SEGMENTATION_POLICY,
	make_task_segments as make_common_task_segments,
	segmentation_policy_metadata,
)


# =========================
# Configuration
# =========================
ROOT = Path("/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis")
DATA_ROOT = ROOT / "NullSpace_Data"
JOINT_ANGLES_ROOT = DATA_ROOT / "Joint_angles_time_series"
CHECKED_ABSREF_ROOT = DATA_ROOT / "Checked_Absref_data"
RESULTS_VARIANT_ROOT = (
	ROOT / "Results_2" / "UCM_more_robust_minimal_mathfix_common_inclusive"
)

@dataclass
class AnalysisConfig:
	name: str
	results_root: Path
	file_suffix: str
	interest_columns: List[str]
	angle_source_map: Dict[str, str]
	# Display column names whose source lives in the absref CSV (not joint_angles CSV)
	absref_angle_columns: List[str] = field(default_factory=list)

	@property
	def data_out_dir(self) -> Path:
		return self.results_root / "data"

	@property
	def fig_out_dir(self) -> Path:
		return self.results_root / "figures"


# 4-DOF: upper limb without wrist
CONFIG_WITHOUT_WRIST = AnalysisConfig(
	name="without_wrist",
	results_root=RESULTS_VARIANT_ROOT / "UCM_four_stage_upper_limb_angle_without_wrist",
	file_suffix="_without_wrist",
	interest_columns=[
		"elbow_flex_deg",
		"shoulder_elev_deg",
		"shoulder_plane_deg",
		"shoulder_axial_proxy_deg",
	],
	angle_source_map={
		"elbow_flex_deg": "elbow_flex_deg",
		"shoulder_elev_deg": "shoulder_elev_deg",
		"shoulder_plane_deg": "shoulder_plane_unwrap_deg",
		"shoulder_axial_proxy_deg": "shoulder_axial_proxy_unwrap_deg",
	},
)

# 7-DOF: full upper limb including wrist
CONFIG_WITH_WRIST = AnalysisConfig(
	name="with_wrist",
	results_root=RESULTS_VARIANT_ROOT / "UCM_four_stage_upper_limb_angle",
	file_suffix="",
	interest_columns=[
		"elbow_flex_deg",
		"shoulder_elev_deg",
		"shoulder_plane_deg",
		"shoulder_axial_proxy_deg",
		"wrist_flex_ext_deg180",
		"wrist_dev_deg",
		"wrist_prosup_deg",
	],
	angle_source_map={
		"elbow_flex_deg": "elbow_flex_deg",
		"shoulder_elev_deg": "shoulder_elev_deg",
		"shoulder_plane_deg": "shoulder_plane_unwrap_deg",
		"shoulder_axial_proxy_deg": "shoulder_axial_proxy_unwrap_deg",
		"wrist_flex_ext_deg180": "wrist_flex_ext_deg180",
		"wrist_dev_deg": "wrist_dev_deg",
		"wrist_prosup_deg": "wrist_prosup_deg",
	},
)

# Thumb CMC 2-DoF proxy columns only (sourced from joint_angles CSV)
_THUMB_CMC_COLUMNS: List[str] = [
	"thumb_cmc_flex_ext_deg",
	"thumb_cmc_abd_add_deg",
]

# Thumb-only joint angle column names (subset of finger columns, from joint_angles CSV)
_THUMB_COLUMNS: List[str] = [
	# THUMB angles (original chain-based)
	"THUMB_cmc_flex_deg",
	"THUMB_mcp_flex_deg",
	"THUMB_ip_flex_deg",
	# Thumb CMC 2-DoF proxies
	"thumb_cmc_flex_ext_deg",
	"thumb_cmc_abd_add_deg",
]

# Finger joint angle column names (sourced from joint_angles CSV)
_FINGER_COLUMNS: List[str] = [
	# INDEX finger
	"INDEX_mcp_flex_deg",
	"INDEX_pip_flex_deg",
	"INDEX_dip_flex_deg",
	# MIDDLE finger
	"MIDDLE_mcp_flex_deg",
	"MIDDLE_pip_flex_deg",
	"MIDDLE_dip_flex_deg",
	# RING finger
	"RING_mcp_flex_deg",
	"RING_pip_flex_deg",
	"RING_dip_flex_deg",
	# PINKY finger
	"PINKY_mcp_flex_deg",
	"PINKY_pip_flex_deg",
	"PINKY_dip_flex_deg",
	# THUMB angles (original chain-based)
	"THUMB_cmc_flex_deg",
	"THUMB_mcp_flex_deg",
	"THUMB_ip_flex_deg",
	# Thumb CMC 2-DoF proxies
	"thumb_cmc_flex_ext_deg",
	"thumb_cmc_abd_add_deg",
]

# 27-DOF: full upper limb + wrist + fingers (sourced from joint_angles CSV)
CONFIG_WITH_FINGERS = AnalysisConfig(
	name="with_fingers",
	results_root=RESULTS_VARIANT_ROOT / "UCM_four_stage_upper_limb_angle_with_fingers",
	file_suffix="_with_fingers",
	interest_columns=[
		"elbow_flex_deg",
		"shoulder_elev_deg",
		"shoulder_plane_deg",
		"shoulder_axial_proxy_deg",
		"wrist_flex_ext_deg180",
		"wrist_dev_deg",
		"wrist_prosup_deg",
		*_FINGER_COLUMNS,
	],
	angle_source_map={
		"elbow_flex_deg": "elbow_flex_deg",
		"shoulder_elev_deg": "shoulder_elev_deg",
		"shoulder_plane_deg": "shoulder_plane_unwrap_deg",
		"shoulder_axial_proxy_deg": "shoulder_axial_proxy_unwrap_deg",
		"wrist_flex_ext_deg180": "wrist_flex_ext_deg180",
		"wrist_dev_deg": "wrist_dev_deg",
		"wrist_prosup_deg": "wrist_prosup_deg",
		**{c: c for c in _FINGER_COLUMNS},
	},
)


# 12-DOF: full upper limb + wrist + thumb only (sourced from joint_angles CSV)
CONFIG_WITH_WRIST_AND_THUMB = AnalysisConfig(
	name="with_wrist_and_thumb",
	results_root=RESULTS_VARIANT_ROOT / "UCM_four_stage_upper_limb_angle_with_thumb",
	file_suffix="_with_thumb",
	interest_columns=[
		"elbow_flex_deg",
		"shoulder_elev_deg",
		"shoulder_plane_deg",
		"shoulder_axial_proxy_deg",
		"wrist_flex_ext_deg180",
		"wrist_dev_deg",
		"wrist_prosup_deg",
		*_THUMB_COLUMNS,
	],
	angle_source_map={
		"elbow_flex_deg": "elbow_flex_deg",
		"shoulder_elev_deg": "shoulder_elev_deg",
		"shoulder_plane_deg": "shoulder_plane_unwrap_deg",
		"shoulder_axial_proxy_deg": "shoulder_axial_proxy_unwrap_deg",
		"wrist_flex_ext_deg180": "wrist_flex_ext_deg180",
		"wrist_dev_deg": "wrist_dev_deg",
		"wrist_prosup_deg": "wrist_prosup_deg",
		**{c: c for c in _THUMB_COLUMNS},
	},
)

# 7-DOF: upper limb + wrist flexion only + thumb CMC proxies only (no chain-based thumb, no wrist_dev, no wrist_prosup)
CONFIG_WITH_WRIST_FLEX_AND_CMC = AnalysisConfig(
	name="with_wrist_flex_and_cmc",
	results_root=RESULTS_VARIANT_ROOT / "UCM_four_stage_upper_limb_angle_with_wrist_flex_and_cmc",
	file_suffix="_with_wrist_flex_and_cmc",
	interest_columns=[
		"elbow_flex_deg",
		"shoulder_elev_deg",
		"shoulder_plane_deg",
		"shoulder_axial_proxy_deg",
		"wrist_flex_ext_deg180",
		*_THUMB_CMC_COLUMNS,
	],
	angle_source_map={
		"elbow_flex_deg": "elbow_flex_deg",
		"shoulder_elev_deg": "shoulder_elev_deg",
		"shoulder_plane_deg": "shoulder_plane_unwrap_deg",
		"shoulder_axial_proxy_deg": "shoulder_axial_proxy_unwrap_deg",
		"wrist_flex_ext_deg180": "wrist_flex_ext_deg180",
		**{c: c for c in _THUMB_CMC_COLUMNS},
	},
)

# 10-DOF: upper limb + wrist flexion only + thumb (no wrist_dev, no wrist_prosup)
CONFIG_WITH_WRIST_FLEX_AND_THUMB = AnalysisConfig(
	name="with_wrist_flex_and_thumb",
	results_root=RESULTS_VARIANT_ROOT / "UCM_four_stage_upper_limb_angle_with_wrist_flex_and_thumb",
	file_suffix="_with_wrist_flex_and_thumb",
	interest_columns=[
		"elbow_flex_deg",
		"shoulder_elev_deg",
		"shoulder_plane_deg",
		"shoulder_axial_proxy_deg",
		"wrist_flex_ext_deg180",
		*_THUMB_COLUMNS,
	],
	angle_source_map={
		"elbow_flex_deg": "elbow_flex_deg",
		"shoulder_elev_deg": "shoulder_elev_deg",
		"shoulder_plane_deg": "shoulder_plane_unwrap_deg",
		"shoulder_axial_proxy_deg": "shoulder_axial_proxy_unwrap_deg",
		"wrist_flex_ext_deg180": "wrist_flex_ext_deg180",
		**{c: c for c in _THUMB_COLUMNS},
	},
)

#ALL_CONFIGS = [CONFIG_WITH_WRIST_FLEX_AND_CMC, CONFIG_WITH_WRIST_FLEX_AND_THUMB, CONFIG_WITHOUT_WRIST, CONFIG_WITH_WRIST, CONFIG_WITH_FINGERS, CONFIG_WITH_WRIST_AND_THUMB]
ALL_CONFIGS = [CONFIG_WITH_WRIST_FLEX_AND_CMC]

TASK_LIST = ["Coins", "Tapes", "Tapes_Standing", "Pegs", "Pegs_Standing", "Cognitive"]
PHASES = ["grasp"]
NORMALIZED_LENGTH = 200
ROBUST_Z_THRESHOLD = 3.5

FS = 60.0
USE_LOWPASS = True
LOWPASS_CUTOFF_HZ = 8.0
LOWPASS_ORDER = 2
PROMINENCE_FACTOR = 0.6
MIN_DISTANCE_S = 0.3
THRESHOLD_LOW = -50.0
THRESHOLD_HIGH = 170.0
MARKER_SEG_COL = "WRIST_y"

# Jacobian window parameters
SHORT_SEGMENT_THRESHOLD = 32
IDEAL_WINDOW_LENGTH = 18
MIN_WINDOW_LENGTH = 14
MAX_WINDOW_LENGTH = 22
# Retained legacy ridge scale for underdetermined local regressions only.
JACOBIAN_REG = 1e-3
TASK_SPACE_DIMENSION = 3
ORTHOGONAL_PARTITION_RELATIVE_TOLERANCE = 1e-10


# =========================
# Utility
# =========================
def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def interp_fill_1d(x: np.ndarray) -> np.ndarray:
	return pd.Series(x, dtype=float).interpolate(limit_direction="both").to_numpy(dtype=float)


def lowpass_safe(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
	x = np.asarray(x, dtype=float)
	if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
		return x
	if np.isnan(x).any():
		x = interp_fill_1d(x)
	b, a = butter(order, cutoff_hz / (fs * 0.5), btype="low")
	padlen = 3 * (max(len(a), len(b)) - 1)
	if x.size <= padlen:
		return x
	try:
		return filtfilt(b, a, x, method="gust")
	except Exception:
		return x


def moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
	x = np.asarray(x, dtype=float)
	if k <= 1:
		return x.copy()
	if k % 2 == 0:
		k += 1
	r = k // 2
	xp = np.pad(x, (r, r), mode="reflect")
	w = np.ones(k, dtype=float) / k
	return np.convolve(xp, w, mode="valid")


def normalize_to_length(arr: np.ndarray, target_length: int = NORMALIZED_LENGTH) -> np.ndarray:
	"""Normalize (T, d) to fixed length with linear interpolation."""
	arr = np.asarray(arr, dtype=float)
	t_orig, d = arr.shape
	if t_orig < 2:
		return np.tile(arr[0], (target_length, 1))
	t0 = np.linspace(0.0, 1.0, t_orig)
	t1 = np.linspace(0.0, 1.0, target_length)
	out = np.zeros((target_length, d), dtype=float)
	for i in range(d):
		f = interp1d(t0, arr[:, i], kind="linear", bounds_error=False, fill_value="extrapolate")
		out[:, i] = f(t1)
	return out


def normalize_jacobians_to_length(j_per_frame: np.ndarray, target_length: int = NORMALIZED_LENGTH) -> np.ndarray:
	"""Put interval Jacobians on the same normalised grid as joint positions.

	``j_per_frame[k]`` is estimated from ``q[k+1] - q[k]`` and therefore lives
	at the interval midpoint, not at either endpoint.  The legacy implementation
	incorrectly placed the first and last interval Jacobians at phases 0 and 1.
	Boundary position frames use the nearest interval Jacobian.
	"""
	j = np.asarray(j_per_frame, dtype=float)
	t_orig, m, n = j.shape
	if t_orig < 1:
		raise ValueError("At least one interval Jacobian is required")
	if t_orig == 1:
		return np.tile(j[0], (target_length, 1, 1))
	# A segment with t_orig intervals has t_orig + 1 position samples.  Interval
	# k is centred at (k + 0.5) / t_orig on the position-sample phase axis.
	t0 = (np.arange(t_orig, dtype=float) + 0.5) / float(t_orig)
	t1 = np.linspace(0.0, 1.0, target_length)
	out = np.zeros((target_length, m, n), dtype=float)
	for r in range(m):
		for c in range(n):
			col = j[:, r, c]
			if np.isnan(col).any():
				col = interp_fill_1d(col)
			f = interp1d(
				t0,
				col,
				kind="linear",
				bounds_error=False,
				fill_value=(float(col[0]), float(col[-1])),
			)
			out[:, r, c] = f(t1)
	return out


def day_from_subj_day(subj_day: str) -> Optional[int]:
	if "_Day" not in subj_day:
		return None
	try:
		return int(subj_day.split("_Day")[-1])
	except Exception:
		return None


def canonical_task_name(task_name: str) -> str:
	if task_name.endswith("_First"):
		return task_name[:-6]
	return task_name


# =========================
# Trial discovery / loading
# =========================
def discover_joint_angle_trials() -> List[Dict[str, Any]]:
	trials: List[Dict[str, Any]] = []
	allowed_tasks = {t.strip() for t in TASK_LIST}
	pattern = "PT_*/PT_*_Day*/**/*_joint_angles.csv"
	for csv_path in sorted(JOINT_ANGLES_ROOT.glob(pattern)):
		trial = csv_path.parent.name
		subj_day = csv_path.parent.parent.name
		subj = csv_path.parent.parent.parent.name
		if "_" not in trial:
			continue
		task, block_str = trial.rsplit("_", 1)
		if not block_str.isdigit():
			continue
		if allowed_tasks and canonical_task_name(task) not in allowed_tasks:
			continue
		day = day_from_subj_day(subj_day)
		if day is None:
			continue
		trials.append(
			{
				"subject": subj,
				"subj_day": subj_day,
				"day": day,
				"trial": trial,
				"task": canonical_task_name(task),
				"block": int(block_str),
				"joint_csv": csv_path,
			}
		)
	return trials


def load_joint_angles(joint_csv: Path) -> Optional[pd.DataFrame]:
	if not joint_csv.exists():
		return None
	try:
		return pd.read_csv(joint_csv)
	except Exception:
		return None


def load_checked_absref(subject: str, subj_day: str, trial: str) -> Optional[pd.DataFrame]:
	file_path = CHECKED_ABSREF_ROOT / subject / subj_day / trial / f"{trial}_with_table_absref.csv"
	if not file_path.exists():
		return None
	try:
		return pd.read_csv(file_path)
	except Exception:
		return None


def pick_angle_source_columns(df_angles: pd.DataFrame, cfg: AnalysisConfig) -> Optional[List[str]]:
	selected: List[str] = []
	for display_col in cfg.interest_columns:
		source_col = cfg.angle_source_map[display_col]
		if source_col not in df_angles.columns:
			return None
		selected.append(source_col)
	return selected


AUGMENTATION_TASKS_USE_TT_TIP = {
	"Tapes",
	"Tapes_Standing",
	"Pegs",
	"Pegs_Standing",
}


def select_endpoint_xyz_columns(df_absref: pd.DataFrame, task: str) -> Optional[List[str]]:
	"""Pick end-effector marker columns by task definition.

	Coins -> THUMB_TIP_* ; augmentation tasks -> THIRD_THUMB_TIP_*
	"""
	task_name = canonical_task_name(task)
	if task_name == "Coins":
		prefix = "THUMB_TIP"
	elif task_name in AUGMENTATION_TASKS_USE_TT_TIP:
		prefix = "THIRD_THUMB_TIP"
	else:
		prefix = "THUMB_TIP"
	cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
	if all(c in df_absref.columns for c in cols):
		return cols
	return None


# =========================
# Four-stage segmentation
# =========================
def segment_four_stage(wrist_y: np.ndarray, fs: float = FS) -> List[Dict[str, Any]]:
	y = np.asarray(wrist_y, dtype=float)
	if np.isnan(y).any():
		y = interp_fill_1d(y)
	y_f = lowpass_safe(y, LOWPASS_CUTOFF_HZ, fs, LOWPASS_ORDER) if USE_LOWPASS else y
	prom = max(float(np.nanstd(y_f)) * PROMINENCE_FACTOR, 1e-6)
	distance = max(int(fs * MIN_DISTANCE_S), 1)
	peaks, _ = find_peaks(y_f, prominence=prom, distance=distance)
	troughs, _ = find_peaks(-y_f, prominence=prom, distance=distance)
	troughs = np.unique(np.r_[0, troughs, len(y_f) - 1]).astype(int)
	troughs.sort()

	segments: List[Dict[str, Any]] = []
	prev_low_down: Optional[int] = None
	for cycle_idx in range(len(troughs) - 1):
		t0 = int(troughs[cycle_idx])
		t1 = int(troughs[cycle_idx + 1])
		peaks_in_cycle = peaks[(peaks > t0) & (peaks < t1)]
		if peaks_in_cycle.size == 0:
			continue
		peak_idx = int(peaks_in_cycle[np.argmax(y_f[peaks_in_cycle])])
		if float(y_f[peak_idx]) < THRESHOLD_HIGH:
			continue

		low_up = next((t0 + j for j, v in enumerate(y_f[t0 : peak_idx + 1]) if v >= THRESHOLD_LOW), None)
		if low_up is None:
			continue
		high_up = next((low_up + j for j, v in enumerate(y_f[low_up : peak_idx + 1]) if v >= THRESHOLD_HIGH), None)
		if high_up is None:
			continue
		high_down = next((peak_idx + j for j, v in enumerate(y_f[peak_idx : t1 + 1]) if v <= THRESHOLD_HIGH), None)
		if high_down is None:
			continue
		low_down = next((high_down + j for j, v in enumerate(y_f[high_down : t1 + 1]) if v <= THRESHOLD_LOW), None)
		if low_down is None:
			continue

		grasp_start = int(prev_low_down) if prev_low_down is not None else t0
		segments.extend(
			[
				{"cycle_idx": cycle_idx, "phase": "grasp", "start": grasp_start, "end": int(low_up)},
				{"cycle_idx": cycle_idx, "phase": "carrying", "start": int(low_up), "end": int(high_up)},
				{"cycle_idx": cycle_idx, "phase": "placement", "start": int(high_up), "end": int(peak_idx)},
				{"cycle_idx": cycle_idx, "phase": "return", "start": int(peak_idx), "end": int(low_down)},
			]
		)
		prev_low_down = int(low_down)

	filtered: List[Dict[str, Any]] = []
	for seg in segments:
		if seg["start"] < seg["end"]:
			filtered.append(seg)
	return filtered


# =========================
# Task variables (NSTS-like)
# =========================
def _phase_waypoint_idx_from_wrist_segment(phase: str, wrist_y_seg: np.ndarray) -> Optional[int]:
	"""Phase-specific waypoint definition following Trajectory_error.py."""
	y = np.asarray(wrist_y_seg, dtype=float)
	if y.ndim != 1 or y.size < 3:
		return None
	if phase == "grasp":
		return int(np.argmin(y))
	if phase in {"placement", "displacement"}:
		return int(np.argmax(y))
	return None


def _project_to_line_signed(pxy: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	v = p1 - p0
	line_len = float(np.linalg.norm(v))
	if line_len < 1e-12:
		s = np.zeros(pxy.shape[0], dtype=float)
		e_n = np.zeros(pxy.shape[0], dtype=float)
		return s, e_n
	t_hat = v / line_len
	n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
	rel = pxy - p0
	s = np.clip(rel @ t_hat, 0.0, line_len)
	e_n = rel @ n_hat
	return s, e_n


def _project_to_polyline_signed(
	pxy: np.ndarray,
	p0: np.ndarray,
	pw: np.ndarray,
	p1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	v1 = pw - p0
	v2 = p1 - pw
	l1 = float(np.linalg.norm(v1))
	l2 = float(np.linalg.norm(v2))
	if l1 < 1e-12 or l2 < 1e-12:
		return _project_to_line_signed(pxy, p0, p1)

	t1 = v1 / l1
	t2 = v2 / l2
	n1 = np.array([-t1[1], t1[0]], dtype=float)
	n2 = np.array([-t2[1], t2[0]], dtype=float)

	s = np.zeros(pxy.shape[0], dtype=float)
	e_n = np.zeros(pxy.shape[0], dtype=float)

	for i, p in enumerate(pxy):
		rel1 = p - p0
		u1 = float(np.clip(rel1 @ t1, 0.0, l1))
		proj1 = p0 + t1 * u1
		dvec1 = p - proj1
		d1 = float(np.linalg.norm(dvec1))
		sign1 = float(np.sign(dvec1 @ n1))

		rel2 = p - pw
		u2 = float(np.clip(rel2 @ t2, 0.0, l2))
		proj2 = pw + t2 * u2
		dvec2 = p - proj2
		d2 = float(np.linalg.norm(dvec2))
		sign2 = float(np.sign(dvec2 @ n2))

		if d1 <= d2:
			s[i] = u1
			e_n[i] = d1 * sign1
		else:
			s[i] = l1 + u2
			e_n[i] = d2 * sign2

	return s, e_n


def _project_to_polyline_monotone_dp(
	pxy: np.ndarray,
	p0: np.ndarray,
	pw: np.ndarray,
	p1: np.ndarray,
	n_bins: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Monotone arc-length projection on polyline p0->pw->p1 via dynamic programming.

	For each frame t, selects an arc-length bin on the polyline such that the
	resulting sequence s[0..T-1] is monotonically non-decreasing while minimising
	the total squared distance from trajectory points to polyline samples.  This
	avoIds backward jumps in `s` that appear near the fold point when each frame
	is projected independently to its nearest polyline segment.
	"""
	v1 = pw - p0
	v2 = p1 - pw
	l1 = float(np.linalg.norm(v1))
	l2 = float(np.linalg.norm(v2))
	if l1 < 1e-12 or l2 < 1e-12:
		return _project_to_line_signed(pxy, p0, p1)

	t1_hat = v1 / l1
	t2_hat = v2 / l2
	n1 = np.array([-t1_hat[1], t1_hat[0]], dtype=float)
	n2 = np.array([-t2_hat[1], t2_hat[0]], dtype=float)
	L = l1 + l2

	# Sample points along the polyline at n_bins arc-length positions
	s_grid = np.linspace(0.0, L, n_bins)
	on_seg1 = s_grid <= l1
	poly_pts = np.where(
		on_seg1[:, np.newaxis],
		p0 + np.outer(np.minimum(s_grid, l1), t1_hat),
		pw + np.outer(np.maximum(s_grid - l1, 0.0), t2_hat),
	)  # (n_bins, 2)

	T = pxy.shape[0]
	if T == 0:
		return np.empty(0, dtype=float), np.empty(0, dtype=float)

	# cost[t, k]: squared distance from trajectory point t to polyline sample k
	diff = pxy[:, np.newaxis, :] - poly_pts[np.newaxis, :, :]  # (T, n_bins, 2)
	cost = np.sum(diff ** 2, axis=2)  # (T, n_bins)

	# DP with monotonicity constraint:
	#   dp[k]     = min cumulative cost ending at bin k at the current time step
	#   dp_new[k] = cost[t, k] + min_{j <= k} dp[j]   (j <= k enforces non-decreasing s)
	# backtrack[t, k] stores argmin_{j <= k} dp[t-1][j] for path reconstruction.
	dp = cost[0].copy()
	backtrack = np.empty((T, n_bins), dtype=np.int32)
	backtrack[0] = np.arange(n_bins, dtype=np.int32)

	for t in range(1, T):
		# Prefix argmin in O(n_bins): argmin_pref[k] = argmin_{j<=k} dp[j]
		argmin_pref = np.empty(n_bins, dtype=np.int32)
		cur_min_val = dp[0]
		cur_min_idx = np.int32(0)
		argmin_pref[0] = cur_min_idx
		for k in range(1, n_bins):
			if dp[k] < cur_min_val:
				cur_min_val = dp[k]
				cur_min_idx = np.int32(k)
			argmin_pref[k] = cur_min_idx
		backtrack[t] = argmin_pref
		dp = cost[t] + np.minimum.accumulate(dp)

	# Traceback: recover globally optimal monotone path
	path = np.empty(T, dtype=np.int32)
	path[T - 1] = int(np.argmin(dp))
	for t in range(T - 2, -1, -1):
		path[t] = backtrack[t + 1][path[t + 1]]

	s_out = s_grid[path]

	# Signed lateral deviation using the normal of the assigned segment
	e_n_out = np.empty(T, dtype=float)
	for t in range(T):
		dvec = pxy[t] - poly_pts[path[t]]
		e_n_out[t] = float(dvec @ (n1 if on_seg1[path[t]] else n2))

	return s_out, e_n_out


def calc_stage_task_vars_from_endpoint(
	endpoint_xyz_seg: np.ndarray,
	phase: str,
	wrist_y_seg: np.ndarray,
	smooth_xy_win: int = 9,
) -> np.ndarray:
	"""Return task state X=(s, e_n, z) for one segment with phase-specific ideal path."""
	p = np.asarray(endpoint_xyz_seg, dtype=float)
	if p.ndim != 2 or p.shape[1] != 3:
		raise ValueError("endpoint_xyz_seg must be (T, 3)")
	if np.isnan(p).any():
		p = pd.DataFrame(p).interpolate(limit_direction="both").to_numpy(dtype=float)

	pxy = p[:, :2].copy()
	z_val = p[:, 2].copy()
	cx = moving_average_1d(pxy[:, 0], k=smooth_xy_win)
	cy = moving_average_1d(pxy[:, 1], k=smooth_xy_win)
	c = np.c_[cx, cy]

	p0 = c[0].astype(float)
	p1 = c[-1].astype(float)
	waypoint_idx = _phase_waypoint_idx_from_wrist_segment(phase, np.asarray(wrist_y_seg, dtype=float))
	if waypoint_idx is not None and 0 < int(waypoint_idx) < (c.shape[0] - 1):
		pw = c[int(waypoint_idx)].astype(float)
		s, e_n = _project_to_polyline_monotone_dp(c, p0, pw, p1)
	else:
		s, e_n = _project_to_line_signed(c, p0, p1)
	return np.column_stack([s, e_n, z_val])


# =========================
# UCM core
# =========================
def split_segment_into_windows_local(seg_len: int) -> List[Tuple[int, int]]:
	vel_len = seg_len - 1
	if vel_len < 1:
		return []
	if seg_len < SHORT_SEGMENT_THRESHOLD:
		return [(0, vel_len - 1)]
	n_windows = max(1, round(vel_len / IDEAL_WINDOW_LENGTH))
	avg = vel_len / n_windows
	if avg < MIN_WINDOW_LENGTH:
		n_windows = max(1, vel_len // MIN_WINDOW_LENGTH)
	elif avg > MAX_WINDOW_LENGTH:
		n_windows = max(1, vel_len // MAX_WINDOW_LENGTH)
	windows: List[Tuple[int, int]] = []
	current = 0
	for i in range(n_windows):
		remaining = vel_len - current
		remaining_windows = n_windows - i
		win_len = remaining // remaining_windows
		end = current + win_len - 1
		if i == n_windows - 1:
			end = vel_len - 1
		windows.append((current, end))
		current = end + 1
	return windows


def estimate_jacobian_lstsq(dX: np.ndarray, dQ: np.ndarray, reg: float = JACOBIAN_REG) -> np.ndarray:
	"""Retain the legacy local temporal-increment Jacobian estimator exactly."""
	t, _ = dX.shape
	_, n = dQ.shape
	if t < n:
		a = dQ.T @ dQ + (reg ** 2) * np.eye(n)
		b = dQ.T @ dX
		try:
			j_t = linalg.solve(a, b, assume_a="pos")
			return j_t.T
		except Exception:
			j, _, _, _ = linalg.lstsq(dQ, dX, lapack_driver="gelsy")
			return j.T
	j, _, _, _ = linalg.lstsq(dQ, dX, lapack_driver="gelsy")
	return j.T


def decompose_segment_local(dX: np.ndarray, dQ: np.ndarray) -> np.ndarray:
	"""Estimate per-frame Jacobian for each window of a segment."""
	vel_len, n_dof = dQ.shape
	j_per_frame = np.full((vel_len, 3, n_dof), np.nan, dtype=float)
	for ws, we in split_segment_into_windows_local(seg_len=vel_len + 1):
		if ws > we:
			continue
		dX_w = dX[ws : we + 1]
		dQ_w = dQ[ws : we + 1]
		if dX_w.shape[0] < 2 or not (np.isfinite(dX_w).all() and np.isfinite(dQ_w).all()):
			continue
		j = estimate_jacobian_lstsq(dX_w, dQ_w)
		if not np.isfinite(j).all():
			continue
		for t in range(ws, we + 1):
			j_per_frame[t] = j
	return j_per_frame


def exact_task_projector(j: np.ndarray) -> Tuple[np.ndarray, int]:
	"""Return the exact joint-space row-space projector of one Jacobian."""
	matrix = np.asarray(j, dtype=float)
	if matrix.ndim != 2 or not np.isfinite(matrix).all():
		raise ValueError("Jacobian must be a finite 2-D matrix")
	_, singular, vt = linalg.svd(matrix, full_matrices=True, check_finite=True)
	if singular.size == 0 or singular[0] <= 0:
		rank = 0
	else:
		tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
		rank = int(np.sum(singular > tolerance))
	rank = min(rank, TASK_SPACE_DIMENSION, matrix.shape[1])
	v_task = vt[:rank, :].T
	p_task = v_task @ v_task.T if rank > 0 else np.zeros((matrix.shape[1], matrix.shape[1]))
	return 0.5 * (p_task + p_task.T), rank


def consensus_projection_matrices(
	jacobians: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, float]:
	"""Return a coordinate-invariant common projector for local repetitions.

	Each repetition-specific Jacobian is first replaced by its joint-space task
	projector.  For a fixed target rank, the leading eigenspace of their mean is
	the orthogonal projector that minimises the summed squared Frobenius distance
	to the individual projectors.  Unlike an elementwise mean of Jacobians, this
	construction is unchanged by an invertible re-expression of any repetition's
	task coordinates (up to numerical rank tolerance).
	"""
	array = np.asarray(jacobians, dtype=float)
	if array.ndim != 3:
		raise ValueError("jacobians must have shape (n_repetitions, task_dim, n_dofs)")

	projectors: List[np.ndarray] = []
	ranks: List[int] = []
	for matrix in array:
		if not np.isfinite(matrix).all():
			continue
		projector, rank = exact_task_projector(matrix)
		projectors.append(projector)
		ranks.append(rank)
	if not projectors:
		raise ValueError("No finite repetition Jacobians are available")

	target_rank = int(np.rint(np.median(ranks)))
	target_rank = int(np.clip(target_rank, 0, array.shape[2]))
	mean_projector = np.mean(np.stack(projectors, axis=0), axis=0)
	mean_projector = 0.5 * (mean_projector + mean_projector.T)
	eigenvalues, eigenvectors = linalg.eigh(mean_projector, check_finite=True)
	order = np.argsort(eigenvalues)[::-1]
	eigenvalues = eigenvalues[order]
	eigenvectors = eigenvectors[:, order]
	v_task = eigenvectors[:, :target_rank]
	p_task = v_task @ v_task.T if target_rank > 0 else np.zeros_like(mean_projector)
	p_task = 0.5 * (p_task + p_task.T)
	p_null = np.eye(array.shape[2]) - p_task
	if 0 < target_rank < array.shape[2]:
		eigengap = float(eigenvalues[target_rank - 1] - eigenvalues[target_rank])
	else:
		eigengap = np.nan
	return (
		p_null,
		p_task,
		target_rank,
		int(min(ranks)),
		int(max(ranks)),
		len(projectors),
		eigengap,
	)


def compute_vucm_vort_delta(Q_all: np.ndarray, J_all: np.ndarray) -> Dict[str, Any]:
	"""Decompose between-repetition variance with consensus local UCMs."""
	Q_all = np.asarray(Q_all, dtype=float)
	J_all = np.asarray(J_all, dtype=float)
	if Q_all.ndim != 3 or J_all.ndim != 4:
		raise ValueError("Q_all and J_all must be (N,T,D) and (N,T,M,D)")
	if Q_all.shape[0] != J_all.shape[0] or Q_all.shape[1] != J_all.shape[1] or Q_all.shape[2] != J_all.shape[3]:
		raise ValueError("Q_all and J_all shapes are inconsistent")
	n_trials, t_norm, n_dofs = Q_all.shape
	if n_trials < 2:
		raise ValueError("At least two repetitions are required")
	q_mean = np.nanmean(Q_all, axis=0)
	v_ucm = np.full(t_norm, np.nan, dtype=float)
	v_ort = np.full(t_norm, np.nan, dtype=float)
	v_total = np.full(t_norm, np.nan, dtype=float)
	delta_v = np.full(t_norm, np.nan, dtype=float)
	delta_v_classic = np.full(t_norm, np.nan, dtype=float)
	partition_error = np.full(t_norm, np.nan, dtype=float)
	partition_relative_error = np.full(t_norm, np.nan, dtype=float)
	frame_rank = np.zeros(t_norm, dtype=int)
	individual_rank_min = np.zeros(t_norm, dtype=int)
	individual_rank_max = np.zeros(t_norm, dtype=int)
	valid_jacobian_count = np.zeros(t_norm, dtype=int)
	consensus_eigengap = np.full(t_norm, np.nan, dtype=float)
	sample_df = n_trials - 1

	for t in range(t_norm):
		(
			p_null,
			p_task,
			r,
			rank_min,
			rank_max,
			n_valid_j,
			eigengap,
		) = consensus_projection_matrices(J_all[:, t, :, :])
		n_null = n_dofs - r
		delta_q = Q_all[:, t, :] - q_mean[t]
		dq_ucm = delta_q @ p_null.T
		dq_ort = delta_q @ p_task.T
		null_ss = float(np.sum(dq_ucm ** 2))
		task_ss = float(np.sum(dq_ort ** 2))
		total_ss = float(np.sum(delta_q ** 2))
		v_ucm[t] = null_ss / (sample_df * n_null) if n_null > 0 else np.nan
		v_ort[t] = task_ss / (sample_df * r) if r > 0 else np.nan
		v_total[t] = total_ss / (sample_df * n_dofs)
		delta_v[t] = v_ucm[t] - v_ort[t]
		delta_v_classic[t] = delta_v[t] / v_total[t] if v_total[t] > 0 else np.nan
		partition_error[t] = null_ss + task_ss - total_ss
		partition_relative_error[t] = (
			abs(partition_error[t]) / total_ss if total_ss > 0 else 0.0
		)
		frame_rank[t] = r
		individual_rank_min[t] = rank_min
		individual_rank_max[t] = rank_max
		valid_jacobian_count[t] = n_valid_j
		consensus_eigengap[t] = eigengap

	if float(np.nanmax(partition_relative_error)) > ORTHOGONAL_PARTITION_RELATIVE_TOLERANCE:
		raise RuntimeError(
			"Exact UCM partition failed: max relative error "
			f"{np.nanmax(partition_relative_error):.3e}"
		)
	metadata_rank = int(np.rint(np.median(frame_rank)))
	return {
		"V_UCM_angle": v_ucm,
		"V_ORT_angle": v_ort,
		"DeltaV_angle": delta_v,
		"V_TOTAL_angle": v_total,
		"DeltaV_classic_normalized": delta_v_classic,
		"partition_error_rad2": partition_error,
		"partition_relative_error": partition_relative_error,
		"frame_rank": frame_rank,
		"individual_rank_min": individual_rank_min,
		"individual_rank_max": individual_rank_max,
		"valid_jacobian_count": valid_jacobian_count,
		"consensus_eigengap": consensus_eigengap,
		"rank": metadata_rank,
		"n_null": n_dofs - metadata_rank,
		"n_dofs": n_dofs,
		"n_trials": n_trials,
	}


# =========================
# Trial processing
# =========================
def process_trial(trial_info: Dict[str, Any], cfg: AnalysisConfig) -> List[Dict[str, Any]]:
	"""Return normalized segment records for one trial."""
	subject = trial_info["subject"]
	subj_day = trial_info["subj_day"]
	trial = trial_info["trial"]

	df_angles = load_joint_angles(trial_info["joint_csv"])
	df_absref = load_checked_absref(subject, subj_day, trial)
	if df_angles is None or df_absref is None:
		return []

	endpoint_cols = select_endpoint_xyz_columns(df_absref, trial_info["task"])
	if endpoint_cols is None:
		return []

	required_absref_cols = [MARKER_SEG_COL, *endpoint_cols]
	if not set(required_absref_cols).issubset(df_absref.columns):
		return []

	t_len = min(len(df_angles), len(df_absref))
	df_angles = df_angles.iloc[:t_len].copy()
	df_absref = df_absref.iloc[:t_len].copy()

	# Build angle array from mixed sources: joint_angles CSV and/or absref CSV
	absref_cols_set = set(cfg.absref_angle_columns)
	q_cols: List[np.ndarray] = []
	for display_col in cfg.interest_columns:
		source_col = cfg.angle_source_map[display_col]
		if display_col in absref_cols_set:
			if source_col not in df_absref.columns:
				return []
			q_cols.append(df_absref[source_col].to_numpy(dtype=float))
		else:
			if source_col not in df_angles.columns:
				return []
			q_cols.append(df_angles[source_col].to_numpy(dtype=float))
	q_deg = np.column_stack(q_cols)
	if np.isnan(q_deg).any():
		q_deg = pd.DataFrame(q_deg).interpolate(limit_direction="both").to_numpy(dtype=float)
	q_rad = np.deg2rad(q_deg)

	wrist_y = df_absref[MARKER_SEG_COL].to_numpy(dtype=float)
	canonical_segments = make_common_task_segments(
		trial_info["task"],
		subject,
		trial_info["day"],
		trial_info["block"],
		df_absref,
		fs=FS,
	)
	segments = [
		{"cycle_idx": index, "phase": "grasp", "start": start, "end": end}
		for index, (start, end) in enumerate(canonical_segments)
	]
	if not segments:
		return []

	endpoint_xyz = df_absref[endpoint_cols].to_numpy(dtype=float)
	if np.isnan(endpoint_xyz).any():
		endpoint_xyz = pd.DataFrame(endpoint_xyz).interpolate(limit_direction="both").to_numpy(dtype=float)

	segment_records: List[Dict[str, Any]] = []
	for seg_idx, seg in enumerate(segments):
		start = max(0, int(seg["start"]))
		end = min(t_len - 1, int(seg["end"]))
		if end - start + 1 < 3:
			continue

		q_seg = q_rad[start : end + 1]
		wrist_y_seg = wrist_y[start : end + 1]
		x_seg = calc_stage_task_vars_from_endpoint(
			endpoint_xyz_seg=endpoint_xyz[start : end + 1],
			phase=seg["phase"],
			wrist_y_seg=wrist_y_seg,
		)
		dX = np.diff(x_seg, axis=0)
		dQ = np.diff(q_seg, axis=0)
		if dQ.shape[0] < 2:
			continue

		j_per_frame = decompose_segment_local(dX, dQ)

		q_angle_norm = normalize_to_length(q_seg, NORMALIZED_LENGTH)
		j_norm = normalize_jacobians_to_length(j_per_frame, NORMALIZED_LENGTH)

		segment_records.append(
			{
				"subject": subject,
				"subj_day": subj_day,
				"day": trial_info["day"],
				"task": trial_info["task"],
				"trial": trial,
				"block": trial_info["block"],
				"endpoint_marker": endpoint_cols[0].rsplit("_", 1)[0],
				"phase": seg["phase"],
				"cycle_idx": int(seg["cycle_idx"]),
				"segment_idx": seg_idx,
				"start": start,
				"end": end,
				"segment_len": end - start + 1,
				"q_angle_norm": q_angle_norm,
				"j_norm": j_norm,
			}
		)
	return segment_records


# =========================
# Save and visualization
# =========================
def build_segment_angle_table(rec: Dict[str, Any], cfg: AnalysisConfig) -> pd.DataFrame:
	t = rec["q_angle_norm"].shape[0]
	df = pd.DataFrame(
		{
			"subject": rec["subject"],
			"day": rec["day"],
			"task": rec["task"],
			"trial": rec["trial"],
			"block": rec["block"],
			"phase": rec["phase"],
			"cycle_idx": rec["cycle_idx"],
			"segment_idx": rec["segment_idx"],
			"norm_frame": np.arange(t),
		}
	)
	for i, display_col in enumerate(cfg.interest_columns):
		joint = display_col.replace("_deg180", "").replace("_deg", "")
		df[f"{joint}_angle_rad"] = rec["q_angle_norm"][:, i]
	return df


def create_metric_line_plot(df_scalar: pd.DataFrame, metric: str, out_file: Path, title: str) -> None:
	if df_scalar.empty:
		return
	fig, ax = plt.subplots(figsize=(12, 8))
	rng = np.random.default_rng(42)
	for day in sorted(df_scalar["day"].unique()):
		vals = df_scalar.loc[df_scalar["day"] == day, metric].to_numpy(dtype=float)
		vals = vals[np.isfinite(vals)]
		if vals.size == 0:
			continue
		x = day + rng.uniform(-0.12, 0.12, size=vals.size)
		ax.scatter(x, vals, alpha=0.28, s=36, color="#1f77b4", edgecolors="none")

	stats_rows: List[Dict[str, float]] = []
	for day in sorted(df_scalar["day"].unique()):
		vals = df_scalar.loc[df_scalar["day"] == day, metric].to_numpy(dtype=float)
		vals = vals[np.isfinite(vals)]
		if vals.size == 0:
			continue
		center = float(np.mean(vals))
		err = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
		stats_rows.append({"x": day, "center": center, "low": center - err, "high": center + err})

	if stats_rows:
		st = pd.DataFrame(stats_rows)
		ax.plot(st["x"], st["center"], "-", color="#1f77b4", linewidth=2.5, alpha=0.85)
		ax.errorbar(
			st["x"],
			st["center"],
			yerr=[st["center"] - st["low"], st["high"] - st["center"]],
			fmt="o",
			color="#1f77b4",
			markersize=7,
			markerfacecolor="white",
			capsize=5,
			linewidth=2,
		)

	ax.set_xlabel("Day")
	ax.set_ylabel(metric)
	ax.set_title(title)
	ax.set_xticks([1, 2, 3, 4, 5])
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(out_file, dpi=220)
	plt.close(fig)


def create_profile_plot(df_profiles: pd.DataFrame, metric: str, out_file: Path, title: str) -> None:
	if df_profiles.empty:
		return
	fig, ax = plt.subplots(figsize=(12, 8))
	colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}
	for day in sorted(df_profiles["day"].unique()):
		df_day = df_profiles[df_profiles["day"] == day]
		if df_day.empty:
			continue
		grp = df_day.groupby("norm_frame")[metric]
		mean = grp.mean()
		sem = grp.std(ddof=1) / np.sqrt(np.maximum(grp.count(), 1))
		x = mean.index.to_numpy(dtype=float)
		y = mean.to_numpy(dtype=float)
		e = sem.fillna(0.0).to_numpy(dtype=float)
		color = colors.get(day, "#1f77b4")
		ax.plot(x, y, color=color, linewidth=2.1, label=f"Day{day}")
		ax.fill_between(x, y - e, y + e, color=color, alpha=0.16)

	ax.set_xlabel("Normalized Frame")
	ax.set_ylabel(metric)
	ax.set_title(title)
	ax.grid(alpha=0.25)
	ax.legend(frameon=False, ncol=3)
	fig.tight_layout()
	fig.savefig(out_file, dpi=220)
	plt.close(fig)


def robust_qc_filter(
	group_recs: List[Dict[str, Any]],
	z_threshold: float = ROBUST_Z_THRESHOLD,
) -> List[Dict[str, Any]]:
	"""Remove outlier trials based on per-joint robust z-score of trajectory deviation.

	For each joint, compute RMS deviation of each trial from the group median trajectory,
	then convert to a robust z-score using median and MAD. If any joint's |z| > threshold,
	the trial is excluded.
	"""
	n_trials = len(group_recs)
	if n_trials < 3:
		return group_recs

	# Stack normalised angle trajectories: (n_trials, T, n_dofs)
	q_all = np.stack([r["q_angle_norm"] for r in group_recs], axis=0)

	# Median trajectory across trials: (T, n_dofs)
	q_median = np.median(q_all, axis=0)

	# Per-trial, per-joint RMS deviation from the median trajectory: (n_trials, n_dofs)
	deviations = np.sqrt(np.mean((q_all - q_median[np.newaxis, :, :]) ** 2, axis=1))

	# Robust z-score per joint
	med_dev = np.median(deviations, axis=0)  # (n_dofs,)
	mad_dev = np.median(np.abs(deviations - med_dev[np.newaxis, :]), axis=0)  # (n_dofs,)
	mad_dev = np.where(mad_dev < 1e-12, 1e-12, mad_dev)
	robust_z = (deviations - med_dev[np.newaxis, :]) / (1.4826 * mad_dev[np.newaxis, :])

	# Exclude trial if ANY joint exceeds threshold
	max_abs_z = np.max(np.abs(robust_z), axis=1)  # (n_trials,)
	keep_mask = max_abs_z <= z_threshold

	filtered = [r for r, keep in zip(group_recs, keep_mask) if keep]
	n_removed = n_trials - len(filtered)
	if n_removed > 0:
		subject = group_recs[0]["subject"]
		day = group_recs[0]["day"]
		task = group_recs[0]["task"]
		phase = group_recs[0]["phase"]
		print(
			f"  [QC] {subject} Day{day} {task} {phase}: "
			f"removed {n_removed}/{n_trials} trials (robust z > {z_threshold})"
		)
	return filtered


def aggregate_and_save(all_segment_records: List[Dict[str, Any]], cfg: AnalysisConfig) -> None:
	ensure_dir(cfg.data_out_dir)
	ensure_dir(cfg.fig_out_dir)

	if not all_segment_records:
		print("[warn] No segment records to save")
		return

	segment_tables = [build_segment_angle_table(rec, cfg) for rec in all_segment_records]
	df_segments = pd.concat(segment_tables, ignore_index=True)
	df_segments.to_csv(cfg.data_out_dir / f"four_stage_segment_angle_timeseries{cfg.file_suffix}.csv", index=False)

	metrics_profiles: List[Dict[str, Any]] = []
	metrics_scalars: List[Dict[str, Any]] = []
	qc_total_before = 0
	qc_total_after = 0
	qc_groups_affected = 0

	group_keys = ["subject", "day", "task", "phase"]
	for key, idx in pd.DataFrame(all_segment_records)[group_keys].drop_duplicates().iterrows():
		_ = key
		subject = idx["subject"]
		day = int(idx["day"])
		task = idx["task"]
		phase = idx["phase"]
		group_recs = [
			r
			for r in all_segment_records
			if r["subject"] == subject and r["day"] == day and r["task"] == task and r["phase"] == phase
		]
		if len(group_recs) < 2:
			continue
		# Robust QC: remove outlier trials before UCM decomposition
		n_before = len(group_recs)
		group_recs = robust_qc_filter(group_recs)
		n_after = len(group_recs)
		qc_total_before += n_before
		qc_total_after += n_after
		if n_after < n_before:
			qc_groups_affected += 1
		if len(group_recs) < 2:
			continue
		q_all = np.stack([r["q_angle_norm"] for r in group_recs], axis=0)
		j_all = np.stack([r["j_norm"] for r in group_recs], axis=0)
		m = compute_vucm_vort_delta(q_all, j_all)
		for t in range(NORMALIZED_LENGTH):
			metrics_profiles.append(
				{
					"subject": subject,
					"day": day,
					"task": task,
					"phase": phase,
					"norm_frame": t,
					"V_UCM_angle": float(m["V_UCM_angle"][t]),
					"V_ORT_angle": float(m["V_ORT_angle"][t]),
					"DeltaV_angle": float(m["DeltaV_angle"][t]),
					"V_TOTAL_angle": float(m["V_TOTAL_angle"][t]),
					"DeltaV_classic_normalized": float(m["DeltaV_classic_normalized"][t]),
					"task_rank_frame": int(m["frame_rank"][t]),
					"n_null_frame": int(m["n_dofs"] - m["frame_rank"][t]),
					"individual_rank_min": int(m["individual_rank_min"][t]),
					"individual_rank_max": int(m["individual_rank_max"][t]),
					"valid_jacobian_count": int(m["valid_jacobian_count"][t]),
					"consensus_eigengap": float(m["consensus_eigengap"][t]),
					"partition_error_rad2": float(m["partition_error_rad2"][t]),
					"partition_relative_error": float(m["partition_relative_error"][t]),
					"rank": m["rank"],
					"n_null": m["n_null"],
					"n_dofs": m["n_dofs"],
					"n_trials": m["n_trials"],
				}
			)

		mean_v_ucm = float(np.nanmean(m["V_UCM_angle"]))
		mean_v_ort = float(np.nanmean(m["V_ORT_angle"]))
		mean_v_total = float(np.nanmean(m["V_TOTAL_angle"]))
		metrics_scalars.append(
			{
				"subject": subject,
				"day": day,
				"task": task,
				"phase": phase,
				"V_UCM_angle": mean_v_ucm,
				"V_ORT_angle": mean_v_ort,
				"DeltaV_angle": float(np.nanmean(m["DeltaV_angle"])),
				"V_TOTAL_angle": mean_v_total,
				"DeltaV_classic_normalized": float(
					np.nanmean(m["DeltaV_classic_normalized"])
				),
				"DeltaV_classic_from_mean_components": (
					(mean_v_ucm - mean_v_ort) / mean_v_total
					if mean_v_total > 0 else np.nan
				),
				"frame_rank_min": int(np.min(m["frame_rank"])),
				"frame_rank_max": int(np.max(m["frame_rank"])),
				"individual_rank_min": int(np.min(m["individual_rank_min"])),
				"individual_rank_max": int(np.max(m["individual_rank_max"])),
				"valid_jacobian_count_min": int(np.min(m["valid_jacobian_count"])),
				"consensus_eigengap_min": float(np.nanmin(m["consensus_eigengap"])),
				"max_partition_error_rad2": float(
					np.nanmax(np.abs(m["partition_error_rad2"]))
				),
				"max_partition_relative_error": float(
					np.nanmax(m["partition_relative_error"])
				),
				"jacobian_combination": "projector_consensus_eigendecomposition",
				"projection_method": "exact_orthogonal",
				"rank": m["rank"],
				"n_null": m["n_null"],
				"n_dofs": m["n_dofs"],
				"n_trials": m["n_trials"],
			}
		)

	df_profiles = pd.DataFrame(metrics_profiles)
	df_scalars = pd.DataFrame(metrics_scalars)
	df_profiles.to_csv(cfg.data_out_dir / f"vucm_vort_deltaV_angle_profiles{cfg.file_suffix}.csv", index=False)
	df_scalars.to_csv(cfg.data_out_dir / f"vucm_vort_deltaV_angle_summary{cfg.file_suffix}.csv", index=False)

	qc_removed = qc_total_before - qc_total_after
	pct = 100.0 * qc_removed / qc_total_before if qc_total_before > 0 else 0.0
	print(
		f"\n[QC summary | {cfg.name}] "
		f"Total segments entering QC: {qc_total_before} | "
		f"Removed: {qc_removed} ({pct:.1f}%) | "
		f"Remaining: {qc_total_after} | "
		f"Groups affected: {qc_groups_affected}"
	)

	meta = pd.DataFrame(
		[
			{
				"analysis_columns_display": "|".join(cfg.interest_columns),
				"analysis_columns_source": "|".join([cfg.angle_source_map[c] for c in cfg.interest_columns]),
				"normalized_length": NORMALIZED_LENGTH,
				"analysis_variant": "trajectory_relative_minimal_mathfix",
				"jacobian_regression": f"legacy_local_temporal_lstsq; ridge_lambda={JACOBIAN_REG}_only_when_n_increments_lt_n_dofs",
				"jacobian_time_alignment": "interval_midpoints_to_position_phase_grid",
				"jacobian_combination": "individual_row_space_projectors_then_consensus_eigendecomposition",
				"projection_method": "exact_orthogonal",
				"delta_v_legacy": "V_UCM_angle_minus_V_ORT_angle",
				"delta_v_classic": "(V_UCM_angle_minus_V_ORT_angle)/V_TOTAL_angle, computed framewise",
				"partition_relative_tolerance": ORTHOGONAL_PARTITION_RELATIVE_TOLERANCE,
				"segmentation_source": "D_kinematic/grasp_kinematic_common.py",
				"segmentation_policy": DEFAULT_SEGMENTATION_POLICY.name,
				"note": f"mode={cfg.name}; 20260901 UCM estimand with shared inclusive segmentation; repetition-specific (s,e_n,z) paths retained; periodic shoulder joints switched to unwrap columns; ideal path uses phase-specific waypoint; endpoint marker: Coins=THUMB_TIP, augmentation tasks (Tapes/Tapes_Standing/Pegs/Pegs_Standing)=THIRD_THUMB_TIP",
			}
		]
	)
	meta.to_csv(cfg.data_out_dir / f"analysis_angle_metadata{cfg.file_suffix}.csv", index=False)
	pd.DataFrame([segmentation_policy_metadata()]).to_csv(
		cfg.data_out_dir / f"segmentation_policy{cfg.file_suffix}.csv",
		index=False,
	)

	if df_scalars.empty or df_profiles.empty:
		print("[warn] Not enough grouped trials to draw VUCM/VORT/DeltaV angle-domain plots")
		return

	for task in sorted(df_scalars["task"].unique()):
		for phase in PHASES:
			df_s = df_scalars[(df_scalars["task"] == task) & (df_scalars["phase"] == phase)]
			df_p = df_profiles[(df_profiles["task"] == task) & (df_profiles["phase"] == phase)]
			if df_s.empty or df_p.empty:
				continue
			create_metric_line_plot(
				df_s,
				"V_UCM_angle",
				cfg.fig_out_dir / f"line_day_V_UCM_angle_{task}_{phase}{cfg.file_suffix}.png",
				f"V_UCM angle-domain across days - {task} {phase}",
			)
			create_metric_line_plot(
				df_s,
				"V_ORT_angle",
				cfg.fig_out_dir / f"line_day_V_ORT_angle_{task}_{phase}{cfg.file_suffix}.png",
				f"V_ORT angle-domain across days - {task} {phase}",
			)
			create_metric_line_plot(
				df_s,
				"DeltaV_angle",
				cfg.fig_out_dir / f"line_day_DeltaV_angle_{task}_{phase}{cfg.file_suffix}.png",
				f"DeltaV angle-domain across days - {task} {phase}",
			)

			create_profile_plot(
				df_p,
				"V_UCM_angle",
				cfg.fig_out_dir / f"profile_V_UCM_angle_{task}_{phase}{cfg.file_suffix}.png",
				f"V_UCM angle-domain profile - {task} {phase}",
			)
			create_profile_plot(
				df_p,
				"V_ORT_angle",
				cfg.fig_out_dir / f"profile_V_ORT_angle_{task}_{phase}{cfg.file_suffix}.png",
				f"V_ORT angle-domain profile - {task} {phase}",
			)
			create_profile_plot(
				df_p,
				"DeltaV_angle",
				cfg.fig_out_dir / f"profile_DeltaV_angle_{task}_{phase}{cfg.file_suffix}.png",
				f"DeltaV angle-domain profile - {task} {phase}",
			)


def run_analysis(trials: List[Dict[str, Any]], cfg: AnalysisConfig) -> None:
	"""Run the full UCM pipeline for one analysis configuration."""
	print(f"\n{'=' * 88}")
	print(f"Mode: {cfg.name}  |  DOFs: {len(cfg.interest_columns)}  ({', '.join(cfg.interest_columns)})")
	print(f"Results root: {cfg.results_root}")
	print("=" * 88)

	ensure_dir(cfg.results_root)

	n_ok = 0
	all_segment_records: List[Dict[str, Any]] = []
	for info in trials:
		tag = f"{info['subject']}/{info['subj_day']}/{info['trial']}"
		print(f"\n[{cfg.name}] Processing: {tag}")
		try:
			recs = process_trial(info, cfg)
			if not recs:
				print("  [skip] Missing required data/columns or no valid segments")
				continue
			all_segment_records.extend(recs)
			n_ok += 1
			print(f"  [ok] segments={len(recs)}")
		except Exception as exc:
			print(f"  [error] {exc}")

	aggregate_and_save(all_segment_records, cfg)

	print(f"\n[{cfg.name}] Done. Successful trials: {n_ok}/{len(trials)}")
	print(f"Saved data to: {cfg.data_out_dir}")
	print(f"Saved figures to: {cfg.fig_out_dir}")


def run_self_tests() -> None:
	"""Fast synthetic checks for invariance and exact variance partitioning."""
	rng = np.random.default_rng(20260829)
	n_repetitions, n_frames, n_dofs = 9, 5, 7
	q_all = rng.normal(scale=0.1, size=(n_repetitions, n_frames, n_dofs))
	base_j = rng.normal(size=(n_frames, TASK_SPACE_DIMENSION, n_dofs))
	j_all = np.empty((n_repetitions, n_frames, TASK_SPACE_DIMENSION, n_dofs))
	for repetition in range(n_repetitions):
		for frame in range(n_frames):
			rotation, _ = np.linalg.qr(rng.normal(size=(TASK_SPACE_DIMENSION, TASK_SPACE_DIMENSION)))
			j_all[repetition, frame] = rotation @ base_j[frame]

	result = compute_vucm_vort_delta(q_all, j_all)
	if float(np.max(result["partition_relative_error"])) > 1e-12:
		raise AssertionError("Synthetic orthogonal partition check failed")
	weighted_total = (
		(n_dofs - result["frame_rank"]) * result["V_UCM_angle"]
		+ result["frame_rank"] * result["V_ORT_angle"]
	) / n_dofs
	if not np.allclose(weighted_total, result["V_TOTAL_angle"], rtol=1e-11, atol=1e-13):
		raise AssertionError("Dimension-weighted total variance identity failed")

	# An arbitrary orthogonal re-expression of each repetition's task coordinates
	# must not change the joint-space UCM result.
	j_reexpressed = np.empty_like(j_all)
	for repetition in range(n_repetitions):
		for frame in range(n_frames):
			rotation, _ = np.linalg.qr(rng.normal(size=(TASK_SPACE_DIMENSION, TASK_SPACE_DIMENSION)))
			j_reexpressed[repetition, frame] = rotation @ j_all[repetition, frame]
	reexpressed = compute_vucm_vort_delta(q_all, j_reexpressed)
	for metric in ("V_UCM_angle", "V_ORT_angle", "V_TOTAL_angle", "DeltaV_angle"):
		if not np.allclose(result[metric], reexpressed[metric], rtol=1e-10, atol=1e-12):
			raise AssertionError(f"Task-coordinate invariance failed for {metric}")

	# Retained legacy estimator smoke check.  This does not claim that the
	# statistically ill-conditioned legacy regression has been regularised.
	dq = rng.normal(size=(20, n_dofs))
	dx = rng.normal(size=(20, TASK_SPACE_DIMENSION))
	j_legacy = estimate_jacobian_lstsq(dx, dq)
	if j_legacy.shape != (TASK_SPACE_DIMENSION, n_dofs) or not np.isfinite(j_legacy).all():
		raise AssertionError("Legacy Jacobian estimator smoke check failed")

	# Constant interval Jacobians must remain constant after midpoint alignment.
	j_constant = np.ones((6, TASK_SPACE_DIMENSION, n_dofs), dtype=float)
	if not np.allclose(normalize_jacobians_to_length(j_constant, 31), 1.0):
		raise AssertionError("Interval-midpoint Jacobian alignment check failed")


def main() -> None:
	unknown = [arg for arg in sys.argv[1:] if arg != "--self-test-only"]
	if unknown:
		raise SystemExit(f"Unknown arguments: {' '.join(unknown)}")
	run_self_tests()
	print("Synthetic consensus-UCM self-tests passed")
	if "--self-test-only" in sys.argv[1:]:
		return

	print("=" * 88)
	print("UCM trajectory-relative analysis with minimal mathematical corrections")
	print("=" * 88)
	print(f"Joint data root: {JOINT_ANGLES_ROOT}")

	trials = discover_joint_angle_trials()
	print(f"Discovered trials: {len(trials)}")

	for cfg in ALL_CONFIGS:
		run_analysis(trials, cfg)

	print("\n" + "=" * 88)
	print("All modes done.")
	print("=" * 88)


if __name__ == "__main__":
	main()
