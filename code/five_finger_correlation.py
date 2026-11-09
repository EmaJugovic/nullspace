#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Within-subject five-finger MCP correlation analysis for all planned task pairs.

Pipeline for each subject/day/task/block:
1) Obtain canonical grasp segments from D_kinematic/grasp_kinematic_common.py.
2) Concatenate grasp samples across all accepted segments in that block.
3) Compute pairwise Pearson correlations among five MCP joints on each
   concatenated phase signal.

Output:
- long table CSV (one row per pair per block-phase)
- wide table CSV (one row per block-phase with 10 pair columns)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


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
PROJECT_ROOT = Path(os.environ.get(
	"NULLSPACE_PROJECT_ROOT",
	"/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis",
))
DATA_ROOT = PROJECT_ROOT / "NullSpace_Data"
ABSREF_ROOT = DATA_ROOT / "Checked_Absref_data"
ANGLE_ROOT = DATA_ROOT / "Joint_angles_time_series"
RESULT_ROOT = Path(os.environ.get(
	"NULLSPACE_FINGER_CORRELATION_RESULTS_DIR",
	str(PROJECT_ROOT / "Results_2" / "E_finger_correlation_all_task_pairs"),
))
OUT_DIR = RESULT_ROOT / "data"

SUBJECT_LIST = [
	"PT_06", "PT_07", "PT_08", "PT_09", "PT_10", "PT_11", "PT_12", "PT_13", "PT_14",
	"PT_15", "PT_16", "PT_17", "PT_18", "PT_19", "PT_20", "PT_21", "PT_22", "PT_23",
	"PT_24", "PT_25", "PT_26",
]

SUBJECTS_WITH_FIRST_NAMING = {"PT_23", "PT_24", "PT_25", "PT_26"}

DAY_LIST = [1, 2, 3, 4, 5]
TASK_LIST = ["Tapes", "Tapes_Standing", "Pegs", "Pegs_Standing", "Coins", "Cognitive"]
TASK_AVAILABLE_DAYS = {"Cognitive": [1, 3, 5]}
PLANNED_TASK_COMPARISONS = [
	("tapes_vs_coins", "Tapes", "Coins"),
	("tapes_vs_tapes_standing", "Tapes", "Tapes_Standing"),
	("tapes_vs_cognitive", "Tapes", "Cognitive"),
	("tapes_vs_pegs", "Tapes", "Pegs"),
]


def available_days_for_task(task: str):
	return TASK_AVAILABLE_DAYS.get(task, DAY_LIST)
BLOCK_LIST = [1, 2, 3, 4, 5]
PHASES = ["grasp"]

# Five MCP joints (thumb + 4 fingers)
MCP_COLS = [
	"THUMB_mcp_flex_deg",
	"INDEX_mcp_flex_deg",
	"MIDDLE_mcp_flex_deg",
	"RING_mcp_flex_deg",
	"PINKY_mcp_flex_deg",
]

MCP_PAIRS = [
	(MCP_COLS[i], MCP_COLS[j])
	for i in range(len(MCP_COLS))
	for j in range(i + 1, len(MCP_COLS))
]

# Segmentation settings
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
MIN_SAMPLES_FOR_CORR = 3


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


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


def resolve_task_name_for_subject(task: str, subject: str) -> str:
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


def make_trial_segments_four_stage(y: np.ndarray, fs: float) -> List[Dict[str, Tuple[int, int]]]:
	"""
	Return trial-wise segmentation with complete four phases.

	Each returned trial contains:
	- grasp:   prev_cycle crossing_low_down -> current crossing_low_up
	- carrying: crossing_low_up -> crossing_high_up
	- placement: crossing_high_up -> peak
	- return:  peak -> crossing_low_down

	The first detected cycle is skipped because it has no valid previous
	return endpoint to define grasp.
	"""
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

	trials: List[Dict[str, Tuple[int, int]]] = []
	prev_crossing_low_down: Optional[int] = None

	for i in range(len(troughs) - 1):
		trough_start = int(troughs[i])
		trough_end = int(troughs[i + 1])

		peaks_in_cycle = peaks[(peaks > trough_start) & (peaks < trough_end)]
		if len(peaks_in_cycle) == 0:
			continue

		peak_idx = int(peaks_in_cycle[np.argmax(y_f[peaks_in_cycle])])
		if y_f[peak_idx] < THRESHOLD_HIGH:
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

		if prev_crossing_low_down is not None:
			trial = {
				"grasp": (int(prev_crossing_low_down), int(crossing_low_up)),
				"carrying": (int(crossing_low_up), int(crossing_high_up)),
				"placement": (int(crossing_high_up), int(peak_idx)),
				"return": (int(peak_idx), int(crossing_low_down)),
			}
			trials.append(trial)

		prev_crossing_low_down = int(crossing_low_down)

	return trials


def corr_pair(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
	mask = np.isfinite(x) & np.isfinite(y)
	n = int(np.sum(mask))
	if n < MIN_SAMPLES_FOR_CORR:
		return np.nan, n
	xv = x[mask]
	yv = y[mask]
	if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
		return np.nan, n
	return float(np.corrcoef(xv, yv)[0, 1]), n


def build_phase_concatenation(
	q_all: np.ndarray,
	trials: List[Dict[str, Tuple[int, int]]],
) -> Dict[str, np.ndarray]:
	phase_chunks: Dict[str, List[np.ndarray]] = {p: [] for p in PHASES}
	n = q_all.shape[0]

	for trial in trials:
		for phase in PHASES:
			s, e = trial[phase]
			s = max(0, s)
			e = min(n - 1, e)
			if e - s + 1 < MIN_SEG_LEN_FRAMES:
				continue
			phase_chunks[phase].append(q_all[s:e + 1, :])

	phase_concat: Dict[str, np.ndarray] = {}
	for phase, chunks in phase_chunks.items():
		if len(chunks) == 0:
			continue
		phase_concat[phase] = np.concatenate(chunks, axis=0)

	return phase_concat


def make_empty_phase_rows(
	subject: str,
	day: int,
	task: str,
	block: int,
	phase: str,
	status: str,
	reason: str,
	n_trials: int = 0,
	n_samples: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
	wide_row: Dict[str, object] = {
		"subject": subject,
		"day": day,
		"task": task,
		"block": block,
		"phase": phase,
		"n_trials": n_trials,
		"n_samples_concatenated": n_samples,
		"data_status": status,
		"missing_reason": reason,
	}

	long_rows: List[Dict[str, object]] = []
	for joint_i, joint_j in MCP_PAIRS:
		pair_name = f"corr__{joint_i}__{joint_j}"
		wide_row[pair_name] = np.nan
		long_rows.append(
			{
				"subject": subject,
				"day": day,
				"task": task,
				"block": block,
				"phase": phase,
				"n_trials": n_trials,
				"n_samples_concatenated": n_samples,
				"joint_1": joint_i,
				"joint_2": joint_j,
				"n_valid_pair_samples": 0,
				"correlation": np.nan,
				"data_status": status,
				"missing_reason": reason,
			}
		)

	return long_rows, wide_row


def make_empty_block_rows(
	subject: str,
	day: int,
	task: str,
	block: int,
	status: str,
	reason: str,
	n_trials: int = 0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
	long_rows: List[Dict[str, object]] = []
	wide_rows: List[Dict[str, object]] = []
	for phase in PHASES:
		phase_long_rows, phase_wide_row = make_empty_phase_rows(
			subject=subject,
			day=day,
			task=task,
			block=block,
			phase=phase,
			status=status,
			reason=reason,
			n_trials=n_trials,
			n_samples=0,
		)
		long_rows.extend(phase_long_rows)
		wide_rows.append(phase_wide_row)
	return long_rows, wide_rows


def compute_block_phase_correlations(
	subject: str,
	day: int,
	task: str,
	block: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
	long_rows: List[Dict[str, object]] = []
	wide_rows: List[Dict[str, object]] = []

	abs_path, ang_path = trial_paths(subject, day, task, block)
	if not abs_path.exists() or not ang_path.exists():
		return make_empty_block_rows(subject, day, task, block, "missing", "missing_input_file")

	try:
		df_abs = pd.read_csv(abs_path)
		df_ang = pd.read_csv(ang_path)
	except Exception:
		return make_empty_block_rows(subject, day, task, block, "missing", "read_error")

	if MARKER_SEG not in df_abs.columns:
		return make_empty_block_rows(subject, day, task, block, "missing", "missing_segmentation_column")
	if any(c not in df_ang.columns for c in MCP_COLS):
		return make_empty_block_rows(subject, day, task, block, "missing", "missing_mcp_columns")

	n = min(len(df_abs), len(df_ang))
	if n < 3:
		return make_empty_block_rows(subject, day, task, block, "missing", "too_few_samples")

	q_all = df_ang[MCP_COLS].to_numpy(dtype=float)[:n, :]

	segments = make_common_task_segments(
		task,
		subject,
		day,
		block,
		df_abs.iloc[:n].reset_index(drop=True),
		fs=FS,
	)
	trials = [{"grasp": (start, end)} for start, end in segments]
	if len(trials) == 0:
		return make_empty_block_rows(subject, day, task, block, "missing", "no_valid_canonical_grasps")

	phase_concat = build_phase_concatenation(q_all, trials)
	if len(phase_concat) == 0:
		return make_empty_block_rows(subject, day, task, block, "missing", "no_phase_samples", n_trials=len(trials))

	for phase in PHASES:
		if phase not in phase_concat:
			phase_long_rows, phase_wide_row = make_empty_phase_rows(
				subject=subject,
				day=day,
				task=task,
				block=block,
				phase=phase,
				status="missing",
				reason="phase_not_available",
				n_trials=len(trials),
				n_samples=0,
			)
			long_rows.extend(phase_long_rows)
			wide_rows.append(phase_wide_row)
			continue
		q_phase = phase_concat[phase]
		n_samples = int(q_phase.shape[0])

		wide_row: Dict[str, object] = {
			"subject": subject,
			"day": day,
			"task": task,
			"block": block,
			"phase": phase,
			"n_trials": len(trials),
			"n_samples_concatenated": n_samples,
			"data_status": "ok",
			"missing_reason": "",
		}

		for i, (joint_i, joint_j) in enumerate(MCP_PAIRS):
			corr, n_valid = corr_pair(
				q_phase[:, MCP_COLS.index(joint_i)],
				q_phase[:, MCP_COLS.index(joint_j)],
			)

			long_rows.append(
				{
					"subject": subject,
					"day": day,
					"task": task,
					"block": block,
					"phase": phase,
					"n_trials": len(trials),
					"n_samples_concatenated": n_samples,
					"joint_1": joint_i,
					"joint_2": joint_j,
					"n_valid_pair_samples": n_valid,
					"correlation": corr,
					"data_status": "ok",
					"missing_reason": "",
				}
			)

			pair_name = f"corr__{joint_i}__{joint_j}"
			wide_row[pair_name] = corr

		wide_rows.append(wide_row)

	return long_rows, wide_rows


def main() -> None:
	ensure_dir(OUT_DIR)

	all_long_rows: List[Dict[str, object]] = []
	all_wide_rows: List[Dict[str, object]] = []

	for task in TASK_LIST:
		print(f"\n=== Task: {task} ===")
		for subject in SUBJECT_LIST:
			for day in available_days_for_task(task):
				for block in BLOCK_LIST:
					long_rows, wide_rows = compute_block_phase_correlations(subject, day, task, block)
					all_long_rows.extend(long_rows)
					all_wide_rows.extend(wide_rows)

					block_status = "ok"
					block_reason = ""
					if len(wide_rows) > 0:
						status_values = {str(row.get("data_status", "")) for row in wide_rows}
						if status_values == {"missing"}:
							block_status = "missing"
							reasons = sorted({str(row.get("missing_reason", "")) for row in wide_rows if row.get("missing_reason", "")})
							block_reason = reasons[0] if len(reasons) == 1 else ";".join(reasons)
						elif "missing" in status_values:
							block_status = "partial"
							reasons = sorted({str(row.get("missing_reason", "")) for row in wide_rows if row.get("missing_reason", "")})
							block_reason = ";".join(reasons)

					status_suffix = f", reason={block_reason}" if block_reason else ""
					print(
						f"[{block_status.upper()}] {subject} Day{day} {task}_{block}: "
						f"{len(wide_rows)} phases, {len(long_rows)} pair rows{status_suffix}"
					)

	if len(all_long_rows) == 0:
		print("[stop] No valid block-phase correlation result was generated.")
		return

	df_long = pd.DataFrame(all_long_rows)
	df_wide = pd.DataFrame(all_wide_rows)

	long_csv = OUT_DIR / "five_finger_mcp_within_subject_phase_correlation_long.csv"
	wide_csv = OUT_DIR / "five_finger_mcp_within_subject_phase_correlation_wide.csv"
	df_long.to_csv(long_csv, index=False)
	df_wide.to_csv(wide_csv, index=False)

	qc_summary = (
		df_long.assign(valid_correlation=df_long["correlation"].notna())
		.groupby(["task", "day", "phase"], as_index=False)
		.agg(
			total_pair_rows=("correlation", "size"),
			valid_pair_rows=("valid_correlation", "sum"),
			n_subjects_scheduled=("subject", "nunique"),
			n_subjects_with_valid_data=(
				"subject",
				lambda x: x[df_long.loc[x.index, "correlation"].notna()].nunique(),
			),
		)
	)
	qc_summary["missing_pair_rows"] = (
		qc_summary["total_pair_rows"] - qc_summary["valid_pair_rows"]
	)
	qc_csv = OUT_DIR / "correlation_data_qc_by_task_day.csv"
	qc_summary.to_csv(qc_csv, index=False)

	missing_summary = (
		df_long[df_long["data_status"] != "ok"]
		.groupby(["task", "missing_reason"], dropna=False, as_index=False)
		.size()
		.rename(columns={"size": "n_pair_rows"})
	)
	missing_csv = OUT_DIR / "correlation_missingness_by_task_reason.csv"
	missing_summary.to_csv(missing_csv, index=False)

	print("\n" + "=" * 80)
	print(f"Done. Long-format CSV: {long_csv}")
	print(f"Done. Wide-format CSV: {wide_csv}")
	print(f"Done. Data QC table: {qc_csv}")
	print(f"Done. Missingness table: {missing_csv}")
	print("=" * 80)


if __name__ == "__main__":
	main()

