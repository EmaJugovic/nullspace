#!/usr/bin/env python3
"""Compute participant-by-day speed-accuracy regression slopes.

For each participant, task, day, marker, and phase, fit the ordinary least
squares model ``traj_error = intercept + slope * speed_peak`` to raw segment
values exported by ``speed_accuracy_trade_off.py``.  The script also recreates
the manuscript Tapes-versus-Coins random-intercept analysis and checks the four
reported Day-1/Day-5 estimates after three-decimal rounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.formula.api as smf


ANALYSIS_ROOT = Path(
    "/group/plasticity/Projects/Null_space_markerless_tracking/NullSpace_Analysis"
)
GROUP_PREFIX = "/group/plasticity"
WINDOWS_GROUP_PREFIX = r"\\cbsu\data\Group\Plasticity"
REQUIRED_COLUMNS = {
    "Subject",
    "Day",
    "Block",
    "Phase",
    "SegmentIndexInBlock",
    "speed_peak",
    "traj_error",
}
PAPER_TASKS = ("Tapes_Seated", "Coin_Task_Seated")
PAPER_ROLES = {
    "Tapes_Seated": "Interaction augmentation",
    "Coin_Task_Seated": "Tool-use control",
}
MANUSCRIPT_TARGETS = {
    ("Tapes_Seated", 1): (0.255, 0.096, 0.414),
    ("Tapes_Seated", 5): (0.144, -0.043, 0.331),
    ("Coin_Task_Seated", 1): (0.003, -0.171, 0.177),
    ("Coin_Task_Seated", 5): (0.011, -0.183, 0.206),
}


def platform_path(path: Path) -> Path:
    text = str(path)
    normalized = text.replace("\\", "/")
    if os.name == "nt" and normalized.startswith(GROUP_PREFIX):
        suffix = normalized[len(GROUP_PREFIX) :].replace("/", "\\")
        return Path(WINDOWS_GROUP_PREFIX + suffix)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def discover_segment_files(source_root: Path) -> list[Path]:
    files = sorted(
        source_root.glob(
            "*/*/phase_*/segment_level_speed_error_pairs_*.csv"
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No segment-level speed/error CSV files were found under {source_root}"
        )
    return files


def read_segment_outputs(source_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    files = discover_segment_files(source_root)
    for path in files:
        relative = path.relative_to(source_root)
        task, marker = relative.parts[:2]
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame.copy()
        frame.insert(0, "Task", task)
        frame.insert(4, "Marker", marker)
        frame["source_file"] = relative.as_posix()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), files


def compute_participant_day_slopes(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["speed_peak"] = pd.to_numeric(data["speed_peak"], errors="coerce")
    data["traj_error"] = pd.to_numeric(data["traj_error"], errors="coerce")
    data = data[
        np.isfinite(data["speed_peak"])
        & np.isfinite(data["traj_error"])
        & (data["speed_peak"] > 0)
    ].copy()

    keys = ["Task", "Subject", "Day", "Marker", "Phase"]
    rows = []
    for key, frame in data.groupby(keys, observed=True, sort=True):
        x = frame["speed_peak"].to_numpy(dtype=float)
        y = frame["traj_error"].to_numpy(dtype=float)
        row = dict(zip(keys, key))
        row["n_segments"] = int(len(frame))
        row["n_unique_peak_speeds"] = int(np.unique(x).size)
        if len(frame) < 3 or np.unique(x).size < 2:
            row.update(
                {
                    "speed_error_slope": np.nan,
                    "intercept": np.nan,
                    "r_value": np.nan,
                    "r_squared": np.nan,
                    "slope_standard_error": np.nan,
                    "slope_p_value": np.nan,
                    "slope_ci_95_low": np.nan,
                    "slope_ci_95_high": np.nan,
                    "fit_status": "not_estimable",
                }
            )
        else:
            fit = stats.linregress(x, y)
            critical = stats.t.ppf(0.975, len(frame) - 2)
            row.update(
                {
                    "speed_error_slope": float(fit.slope),
                    "intercept": float(fit.intercept),
                    "r_value": float(fit.rvalue),
                    "r_squared": float(fit.rvalue**2),
                    "slope_standard_error": float(fit.stderr),
                    "slope_p_value": float(fit.pvalue),
                    "slope_ci_95_low": float(fit.slope - critical * fit.stderr),
                    "slope_ci_95_high": float(fit.slope + critical * fit.stderr),
                    "fit_status": "ok",
                }
            )
        row["model_formula"] = "traj_error ~ 1 + speed_peak"
        row["slope_units"] = "seconds_if_error_mm_and_speed_mm_per_s"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def make_task_day_descriptives(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = slopes[np.isfinite(slopes["speed_error_slope"])].copy()
    for (task, day, marker, phase), frame in valid.groupby(
        ["Task", "Day", "Marker", "Phase"], observed=True
    ):
        values = frame["speed_error_slope"].to_numpy(dtype=float)
        sem = stats.sem(values) if len(values) > 1 else np.nan
        low, high = (
            stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=sem)
            if len(values) > 1
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "Task": task,
                "Day": int(day),
                "Marker": marker,
                "Phase": phase,
                "n_participants": int(len(values)),
                "mean_speed_error_slope": float(np.mean(values)),
                "sd_speed_error_slope": float(np.std(values, ddof=1)),
                "sem_speed_error_slope": float(sem),
                "ci_95_low_unmodelled": float(low),
                "ci_95_high_unmodelled": float(high),
            }
        )
    return pd.DataFrame(rows).sort_values(["Task", "Day", "Marker", "Phase"])


def check_upstream_endpoint(
    slopes: pd.DataFrame, endpoint_path: Path
) -> tuple[pd.DataFrame, float]:
    endpoint = pd.read_csv(endpoint_path)
    keys = ["Task", "Subject", "Day", "Marker", "Phase"]
    required = set(keys + ["speed_error_slope"])
    missing = sorted(required.difference(endpoint.columns))
    if missing:
        raise ValueError(f"{endpoint_path} is missing required columns: {missing}")
    expected = endpoint[keys + ["speed_error_slope"]].rename(
        columns={"speed_error_slope": "upstream_speed_error_slope"}
    )
    observed = slopes[keys + ["speed_error_slope"]].rename(
        columns={"speed_error_slope": "recomputed_speed_error_slope"}
    )
    check = expected.merge(observed, on=keys, how="outer", indicator=True)
    check["absolute_difference"] = (
        check["upstream_speed_error_slope"]
        - check["recomputed_speed_error_slope"]
    ).abs()
    both_nan = check["upstream_speed_error_slope"].isna() & check[
        "recomputed_speed_error_slope"
    ].isna()
    check["matches_within_1e-10"] = (
        (check["_merge"] == "both")
        & (both_nan | (check["absolute_difference"] <= 1e-10))
    )
    finite_diff = check["absolute_difference"].dropna()
    maximum = float(finite_diff.max()) if not finite_diff.empty else 0.0
    return check, maximum


def paper_model(
    slopes: pd.DataFrame, marker: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    current = slopes[
        slopes["Task"].isin(PAPER_TASKS)
        & (slopes["Marker"] == marker)
        & np.isfinite(slopes["speed_error_slope"])
    ].copy()
    if current.empty:
        raise ValueError(f"No finite paper-model slopes were found for marker {marker}")

    cell = current.groupby(["Task", "Day"], observed=True)[
        "speed_error_slope"
    ].agg(cell_mean="mean", cell_sd="std")
    screened = current.join(cell, on=["Task", "Day"])
    screened["lower_3sd"] = screened["cell_mean"] - 3 * screened["cell_sd"]
    screened["upper_3sd"] = screened["cell_mean"] + 3 * screened["cell_sd"]
    screened["is_outlier_3sd"] = (
        screened["cell_sd"].notna()
        & (
            (screened["speed_error_slope"] < screened["lower_3sd"])
            | (screened["speed_error_slope"] > screened["upper_3sd"])
        )
    )
    model_data = screened[~screened["is_outlier_3sd"]].copy()
    model_data["Day"] = pd.Categorical(
        model_data["Day"], categories=sorted(model_data["Day"].unique()), ordered=True
    )
    model_data["Task"] = pd.Categorical(
        model_data["Task"], categories=list(PAPER_TASKS), ordered=True
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.mixedlm(
            "speed_error_slope ~ C(Day) * C(Task)",
            model_data,
            groups=model_data["Subject"],
        ).fit(reml=True, method="powell", maxiter=1000, disp=False)

    prediction = pd.DataFrame(
        [
            {"Task": task, "Day": day}
            for task in PAPER_TASKS
            for day in model_data["Day"].cat.categories
        ]
    )
    prediction["Task"] = pd.Categorical(
        prediction["Task"], categories=model_data["Task"].cat.categories, ordered=True
    )
    prediction["Day"] = pd.Categorical(
        prediction["Day"], categories=model_data["Day"].cat.categories, ordered=True
    )
    design = np.asarray(
        patsy.build_design_matrices(
            [fit.model.data.design_info], prediction, return_type="dataframe"
        )[0]
    )
    beta = fit.fe_params.to_numpy(dtype=float)
    covariance = fit.cov_params().loc[
        fit.fe_params.index, fit.fe_params.index
    ].to_numpy(dtype=float)
    estimate = design @ beta
    standard_error = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))

    # lmerTest/emmeans used Satterthwaite df in the manuscript.  statsmodels
    # does not expose that approximation, so residual df is used here.  For
    # these data it reproduces all published values and CI limits at 3 d.p.
    residual_df = int(len(model_data) - np.linalg.matrix_rank(fit.model.exog))
    critical = float(stats.t.ppf(0.975, residual_df))
    emmeans = prediction.copy()
    emmeans["task_role"] = emmeans["Task"].map(PAPER_ROLES).astype(str)
    emmeans["estimate"] = estimate
    emmeans["standard_error"] = standard_error
    emmeans["df_for_ci"] = residual_df
    emmeans["ci_95_low"] = estimate - critical * standard_error
    emmeans["ci_95_high"] = estimate + critical * standard_error
    emmeans["model"] = (
        "REML Gaussian mixed model: speed_error_slope ~ categorical Day * Task "
        "+ (1 | Subject), after Task-by-Day mean +/- 3 SD screening"
    )
    emmeans["ci_note"] = (
        "Residual-df t CI in Python; reproduces manuscript Satterthwaite "
        "lmerTest/emmeans results at 3 decimals"
    )

    checks = []
    for (task, day), target in MANUSCRIPT_TARGETS.items():
        row = emmeans[(emmeans["Task"] == task) & (emmeans["Day"] == day)]
        if len(row) != 1:
            raise RuntimeError(f"Missing manuscript cell: {task}, Day {day}")
        row = row.iloc[0]
        calculated = (row["estimate"], row["ci_95_low"], row["ci_95_high"])
        rounded = tuple(float(np.round(value, 3)) for value in calculated)
        checks.append(
            {
                "Task": task,
                "task_role": PAPER_ROLES[task],
                "Day": day,
                "calculated_estimate": calculated[0],
                "calculated_ci_95_low": calculated[1],
                "calculated_ci_95_high": calculated[2],
                "reported_estimate": target[0],
                "reported_ci_95_low": target[1],
                "reported_ci_95_high": target[2],
                "matches_reported_at_3dp": rounded == target,
            }
        )
    check_frame = pd.DataFrame(checks)

    diagnostics = {
        "converged": bool(fit.converged),
        "n_input": int(len(current)),
        "n_outliers_3sd": int(screened["is_outlier_3sd"].sum()),
        "n_model": int(len(model_data)),
        "n_subjects": int(model_data["Subject"].nunique()),
        "fixed_effect_rank": int(np.linalg.matrix_rank(fit.model.exog)),
        "residual_df_for_python_ci": residual_df,
        "random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
        "residual_variance": float(fit.scale),
        "optimizer": "powell",
        "warnings": [str(item.message) for item in caught],
    }
    return screened, model_data, emmeans, check_frame, diagnostics


def sync_files(files: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in files:
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ANALYSIS_ROOT / "Results_2" / "SAT_speed_accuracy_tradeoff",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ANALYSIS_ROOT / "Results_2" / "SAT_speed_accuracy_slope",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ANALYSIS_ROOT / "Results_paper" / "03_speed_accuracy",
    )
    parser.add_argument("--paper-marker", default="WRIST")
    parser.add_argument(
        "--no-paper-sync",
        action="store_true",
        help="Write Results_2 outputs only.",
    )
    parser.add_argument(
        "--allow-manuscript-mismatch",
        action="store_true",
        help="Do not fail if current data no longer reproduce the four reported cells.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = platform_path(args.source_root)
    output_root = platform_path(args.output_root)
    paper_root = platform_path(args.paper_root)

    raw, source_files = read_segment_outputs(source_root)
    slopes = compute_participant_day_slopes(raw)
    descriptives = make_task_day_descriptives(slopes)

    endpoint_path = source_root / "planned_pairwise" / "sat_subject_day_endpoints.csv"
    endpoint_check, maximum_difference = check_upstream_endpoint(slopes, endpoint_path)
    if not endpoint_check["matches_within_1e-10"].all():
        raise RuntimeError(
            "Recomputed slopes do not match the upstream endpoint file; "
            f"maximum finite difference={maximum_difference:.3g}"
        )

    screened, model_data, emmeans, manuscript_check, diagnostics = paper_model(
        slopes, args.paper_marker
    )

    outputs = {
        "slopes": output_root / "participant_day_speed_accuracy_slopes.csv",
        "descriptives": output_root / "task_day_speed_accuracy_slope_descriptives.csv",
        "endpoint_check": output_root / "upstream_slope_reproducibility_check.csv",
        "paper_screening": output_root / "paper_tapes_vs_coins_3sd_screening.csv",
        "paper_model_data": output_root / "paper_tapes_vs_coins_model_data.csv",
        "paper_emmeans": output_root / "paper_tapes_vs_coins_emmeans.csv",
        "manuscript_check": output_root / "manuscript_reported_values_check.csv",
    }
    for frame, path in (
        (slopes, outputs["slopes"]),
        (descriptives, outputs["descriptives"]),
        (endpoint_check, outputs["endpoint_check"]),
        (screened, outputs["paper_screening"]),
        (model_data, outputs["paper_model_data"]),
        (emmeans, outputs["paper_emmeans"]),
        (manuscript_check, outputs["manuscript_check"]),
    ):
        atomic_csv(frame, path)

    provenance_path = output_root / "speed_accuracy_slope_provenance.json"
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_name": "participant_day_speed_error_slope",
        "metric_is_slope": True,
        "participant_day_formula": "traj_error = intercept + slope * speed_peak",
        "paper_analysis": {
            "comparison": "Tapes_Seated versus Coin_Task_Seated",
            "roles": PAPER_ROLES,
            "screening": "mean +/- 3 SD within Task-by-Day cells",
            "model": (
                "Gaussian random-intercept REML model with categorical Day, Task, "
                "and Day-by-Task interaction"
            ),
            "historical_exact_ci_method": "lmerTest/emmeans Satterthwaite df",
            "python_ci_method": "t interval with residual df",
            "diagnostics": diagnostics,
            "all_four_reported_cells_match_at_3dp": bool(
                manuscript_check["matches_reported_at_3dp"].all()
            ),
        },
        "upstream_endpoint_check": {
            "path": str(endpoint_path),
            "sha256": sha256(endpoint_path),
            "all_rows_match_within_1e-10": bool(
                endpoint_check["matches_within_1e-10"].all()
            ),
            "maximum_absolute_difference": maximum_difference,
        },
        "source_analysis": str(
            source_root.parent.parent
            / "Result_analysis_2"
            / "D_kinematic"
            / "speed_accuracy_trade_off.py"
        ),
        "source_root": str(source_root),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_files
        ],
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "outputs": {name: str(path) for name, path in outputs.items()},
        "paper_sync": None,
    }
    atomic_json(provenance, provenance_path)

    if (
        not args.allow_manuscript_mismatch
        and not manuscript_check["matches_reported_at_3dp"].all()
    ):
        raise RuntimeError(
            "Current inputs do not reproduce all four manuscript values at 3 decimals. "
            "Inspect manuscript_reported_values_check.csv."
        )

    if not args.no_paper_sync:
        paper_data_dir = paper_root / "data" / "speed_accuracy_slope"
        synced = sync_files([*outputs.values(), provenance_path], paper_data_dir)
        paper_code_dir = paper_root / "code"
        synced.extend(sync_files([Path(__file__).resolve()], paper_code_dir))
        provenance["paper_sync"] = [str(path) for path in synced]
        atomic_json(provenance, provenance_path)
        shutil.copy2(provenance_path, paper_data_dir / provenance_path.name)

    reported = emmeans[
        emmeans.apply(lambda row: (str(row["Task"]), int(row["Day"])) in MANUSCRIPT_TARGETS, axis=1)
    ]
    print(f"Read {len(source_files)} source CSV files and {len(raw)} segment rows.")
    print(f"Computed {int(np.isfinite(slopes['speed_error_slope']).sum())} participant-day slopes.")
    print(f"Maximum difference from upstream slopes: {maximum_difference:.3g}")
    print("Manuscript cells:")
    print(
        reported[
            ["task_role", "Day", "estimate", "ci_95_low", "ci_95_high"]
        ].to_string(index=False)
    )
    print(f"Results: {output_root}")
    if not args.no_paper_sync:
        print(f"Paper copy: {paper_root / 'data' / 'speed_accuracy_slope'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
