#!/usr/bin/env python3
"""
Compare groups of observed Ripley curves via permutation of mean-curve difference statistics.

Two designs (both common for spatial / functional summary functions):

1. Unpaired two-sample permutation (set_name vs a reference set, e.g. 15F1):
   Curves come from different tomograms, so there is no natural pairing. Shuffle group
   labels across all curves (preserving group sizes) and recompute the chosen statistic.

2. Paired permutation (monomer vs dimer on the same cleft):
   Within each cleft, randomly swap the two class labels (equivalently, randomly flip the
   sign of that cleft's difference curve), then recompute the statistic.

Two statistics were previously supported; analyses now use only:

- integrated_sq_diff: T = ∫ δ(r)² dr
  (any shape difference; no cancellation across r)

This is distinct from the MAD tests in ripley_library.py, which compare one observed curve
to Monte Carlo null replicates within a zone. Here we compare two collections of observed
curves to each other.

Default: 1000 permutations, metrics k_combined & l_combined, windows 0–100 nm and 0–500 nm.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CURVE_ID_COLS = ("tomogram_name", "alignment_dir", "cleft_name")
DEFAULT_METRICS = ("k_combined", "l_combined")
DEFAULT_R_RANGES = (
    ("0-100", 0.0, 100.0),
    ("0-200", 0.0, 200.0),
    ("0-500", 0.0, 500.0),
)
DEFAULT_N_PERM = 1000
DEFAULT_SEED = 42
REFERENCE_SET = "15F1"
FUSING_CURVE_TYPE = "fusing_per_vesicle"

DATASET_CONFIGS = {
    "cleft_center": {
        "sets_csv": "aunp_vs_az_center_ripley_l12_curves.csv",
        "monomer_dimer_csv": "aunp_vs_az_center_ripley_l12_curves_monomer_dimer.csv",
        "metrics": ("k_combined", "l_combined"),
        "label_col": "aunp_pick_label",
        "curve_type_filter": None,
        "extra_id_cols": (),
        "long_family_format": False,
        "out_prefix": "cleft_center",
    },
    "fusion_sites": {
        # Bidirectional long-form export: family in {l12,l21,l_combined,g12,...}, value column.
        # Observed fusing curves only; K families are not stored in this export.
        "sets_csv": "fusion_point_aunp_ripley_bidirectional_curves.csv",
        "monomer_dimer_csv": "fusion_point_aunp_ripley_bidirectional_curves_monomer_dimer.csv",
        "metrics": ("l_combined",),
        "label_col": "aunp_subset",
        "curve_type_filter": "fusing",
        "extra_id_cols": ("replicate_index",),
        "long_family_format": True,
        "out_prefix": "fusion_sites",
    },
}


def _curve_id_cols(
    df: pd.DataFrame,
    *,
    label_col: str | None = None,
    extra_id_cols: tuple[str, ...] = (),
) -> list[str]:
    """Zone/vesicle identity; include class label when monomer/dimer curves coexist."""
    cols = list(CURVE_ID_COLS)
    for c in extra_id_cols:
        if c in df.columns and c not in cols:
            cols.append(c)
    use_label = label_col if label_col and label_col in df.columns else None
    if use_label is None and "aunp_pick_label" in df.columns:
        use_label = "aunp_pick_label"
    if use_label is not None:
        labels = {
            str(x).strip()
            for x in df[use_label].dropna().unique()
            if str(x).strip() and str(x).strip().lower() != "nan"
        }
        # Only treat as a curve-id dimension when ≥2 class labels are present.
        if len(labels) >= 2:
            if use_label not in cols:
                cols.append(use_label)
    return cols


def extract_curves_matrix(
    df: pd.DataFrame,
    value_col: str,
    *,
    id_cols: tuple[str, ...] | list[str] | None = None,
    label_col: str | None = None,
    extra_id_cols: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Pivot long curves to (r_vals, curves[n, n_r], meta[n])."""
    if df.empty or value_col not in df.columns:
        return np.array([]), np.empty((0, 0)), pd.DataFrame()

    sub = df.copy()
    use_cols = (
        list(id_cols)
        if id_cols is not None
        else _curve_id_cols(sub, label_col=label_col, extra_id_cols=extra_id_cols)
    )
    for col in use_cols:
        if col not in sub.columns:
            sub[col] = ""

    r_vals = np.sort(sub["r_nm"].unique().astype(float))
    n_r = len(r_vals)
    r_index = {round(float(r), 6): i for i, r in enumerate(r_vals)}

    curves: list[np.ndarray] = []
    metas: list[dict] = []
    meta_keep = [
        c
        for c in (
            "set_name",
            "aunp_pick_label",
            "aunp_subset",
            "n_aunp_partners",
            "n_query_points",
            "window_volume_nm3",
            "window_mode",
            "curve_type",
            "replicate_index",
        )
        if c in sub.columns and c not in use_cols
    ]

    for keys, grp in sub.groupby(use_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        curve = np.full(n_r, np.nan, dtype=float)
        for r_nm, value in zip(
            grp["r_nm"].to_numpy(dtype=float),
            grp[value_col].to_numpy(dtype=float),
        ):
            idx = r_index.get(round(float(r_nm), 6))
            if idx is not None:
                curve[idx] = value
        if not np.any(np.isfinite(curve)):
            continue
        meta = {col: keys[i] for i, col in enumerate(use_cols)}
        row0 = grp.iloc[0]
        for c in meta_keep:
            meta[c] = row0[c]
        curves.append(curve)
        metas.append(meta)

    if not curves:
        return r_vals, np.empty((0, n_r)), pd.DataFrame()
    return r_vals, np.vstack(curves), pd.DataFrame(metas)


def _r_mask(r_vals: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    return (r_vals >= float(r_min)) & (r_vals <= float(r_max))


STAT_SIGNED_AUC = "signed_auc"  # retained for CSV compatibility / optional re-enable
STAT_INTEGRATED_SQ_DIFF = "integrated_sq_diff"
STATISTIC_NAMES = (STAT_INTEGRATED_SQ_DIFF,)


def _mean_diff_on_window(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_window, δ(r) = mean_A − mean_B) with nonfinite bins dropped."""
    mask = _r_mask(r_vals, r_min, r_max)
    if not np.any(mask):
        return np.array([]), np.array([])
    r = r_vals[mask]
    mean_a = np.nanmean(curves_a[:, mask], axis=0)
    mean_b = np.nanmean(curves_b[:, mask], axis=0)
    diff = mean_a - mean_b
    ok = np.isfinite(diff) & np.isfinite(r)
    return r[ok], diff[ok]


def mean_diff_statistics(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> dict[str, float]:
    """Compute signed AUC and integrated squared difference of mean curves."""
    r, diff = _mean_diff_on_window(curves_a, curves_b, r_vals, r_min=r_min, r_max=r_max)
    if len(r) < 2:
        return {
            "t_signed_auc": float("nan"),
            "t_signed_auc_abs": float("nan"),
            "t_integrated_sq_diff": float("nan"),
        }
    signed = float(np.trapezoid(diff, r))
    sq = float(np.trapezoid(diff * diff, r))
    return {
        "t_signed_auc": signed,
        "t_signed_auc_abs": abs(signed),
        "t_integrated_sq_diff": sq,
    }


def integrated_mean_curve(
    curves: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> float:
    """∫ mean(r) dr over [r_min, r_max]."""
    mask = _r_mask(r_vals, r_min, r_max)
    if not np.any(mask):
        return float("nan")
    r = r_vals[mask]
    mean = np.nanmean(curves[:, mask], axis=0)
    ok = np.isfinite(mean) & np.isfinite(r)
    if ok.sum() < 2:
        return float("nan")
    return float(np.trapezoid(mean[ok], r[ok]))


def _mc_p(t_obs: float, t_null: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(t_null)
    n_ok = int(finite.sum())
    if not np.isfinite(t_obs) or n_ok == 0:
        return float("nan"), n_ok
    p = float((1 + np.sum(t_null[finite] >= t_obs)) / (1 + n_ok))
    return p, n_ok


def unpaired_permutation_test(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    n_perm: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Two-sample permutation test on integrated squared difference of mean curves."""
    n_a = int(curves_a.shape[0])
    n_b = int(curves_b.shape[0])
    obs = mean_diff_statistics(curves_a, curves_b, r_vals, r_min=r_min, r_max=r_max)

    all_curves = np.vstack([curves_a, curves_b])
    n_all = n_a + n_b
    null_sq = np.full(n_perm, np.nan, dtype=float)

    for i in range(n_perm):
        order = rng.permutation(n_all)
        stats_i = mean_diff_statistics(
            all_curves[order[:n_a]],
            all_curves[order[n_a:]],
            r_vals,
            r_min=r_min,
            r_max=r_max,
        )
        null_sq[i] = stats_i["t_integrated_sq_diff"]

    p_sq, n_ok_sq = _mc_p(obs["t_integrated_sq_diff"], null_sq)
    return [
        {
            "design": "unpaired_two_sample",
            "n_a": n_a,
            "n_b": n_b,
            "n_pairs": np.nan,
            "integral_mean_a": integrated_mean_curve(curves_a, r_vals, r_min=r_min, r_max=r_max),
            "integral_mean_b": integrated_mean_curve(curves_b, r_vals, r_min=r_min, r_max=r_max),
            "n_perm": int(n_perm),
            "statistic": STAT_INTEGRATED_SQ_DIFF,
            "t_obs_signed": np.nan,
            "t_obs": obs["t_integrated_sq_diff"],
            "p_value": p_sq,
            "n_perm_finite": n_ok_sq,
            "note": "t_obs = ∫(mean_A − mean_B)² dr (unsigned; no direction)",
        }
    ]


def paired_permutation_test(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    n_perm: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Paired permutation (sign flip) on integrated squared difference of mean curves."""
    if curves_a.shape != curves_b.shape:
        raise ValueError(
            f"Paired curves must match shape; got {curves_a.shape} vs {curves_b.shape}"
        )
    n_pairs = int(curves_a.shape[0])
    diffs = curves_a - curves_b  # (n_pairs, n_r)
    mask = _r_mask(r_vals, r_min, r_max)
    r_full = r_vals[mask]

    def _sq_from_mean_diff(mean_diff: np.ndarray) -> float:
        ok = np.isfinite(mean_diff) & np.isfinite(r_full)
        if ok.sum() < 2:
            return float("nan")
        return float(np.trapezoid(mean_diff[ok] ** 2, r_full[ok]))

    mean_diff_obs = np.nanmean(diffs[:, mask], axis=0)
    t_sq = _sq_from_mean_diff(mean_diff_obs)

    null_sq = np.full(n_perm, np.nan, dtype=float)
    for i in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_pairs)
        mean_diff = np.nanmean((diffs * signs[:, None])[:, mask], axis=0)
        null_sq[i] = _sq_from_mean_diff(mean_diff)

    p_sq, n_ok_sq = _mc_p(t_sq, null_sq)
    return [
        {
            "design": "paired_sign_flip",
            "n_a": n_pairs,
            "n_b": n_pairs,
            "n_pairs": n_pairs,
            "integral_mean_a": integrated_mean_curve(curves_a, r_vals, r_min=r_min, r_max=r_max),
            "integral_mean_b": integrated_mean_curve(curves_b, r_vals, r_min=r_min, r_max=r_max),
            "n_perm": int(n_perm),
            "statistic": STAT_INTEGRATED_SQ_DIFF,
            "t_obs_signed": np.nan,
            "t_obs": t_sq,
            "p_value": p_sq,
            "n_perm_finite": n_ok_sq,
            "note": "t_obs = ∫ [mean(A − B)]² dr (unsigned; no direction)",
        }
    ]

def align_paired_curves(
    df: pd.DataFrame,
    value_col: str,
    *,
    label_col: str = "aunp_pick_label",
    label_a: str = "monomer",
    label_b: str = "dimer",
    extra_id_cols: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (r_vals, curves_a, curves_b, pair_meta) for units with both labels."""
    r_vals, curves, meta = extract_curves_matrix(
        df, value_col, label_col=label_col, extra_id_cols=extra_id_cols
    )
    if curves.shape[0] == 0:
        return r_vals, np.empty((0, 0)), np.empty((0, 0)), pd.DataFrame()

    meta = meta.copy()
    meta["_row"] = np.arange(len(meta))
    pair_keys = [c for c in list(CURVE_ID_COLS) + list(extra_id_cols) if c in meta.columns]
    if label_col not in meta.columns:
        return r_vals, np.empty((0, len(r_vals))), np.empty((0, len(r_vals))), pd.DataFrame()

    wide = meta.pivot_table(
        index=pair_keys,
        columns=label_col,
        values="_row",
        aggfunc="first",
    )
    if label_a not in wide.columns or label_b not in wide.columns:
        return r_vals, np.empty((0, len(r_vals))), np.empty((0, len(r_vals))), pd.DataFrame()

    both = wide[[label_a, label_b]].dropna()
    if both.empty:
        return r_vals, np.empty((0, len(r_vals))), np.empty((0, len(r_vals))), pd.DataFrame()

    idx_a = both[label_a].astype(int).to_numpy()
    idx_b = both[label_b].astype(int).to_numpy()
    pair_meta = both.reset_index()[pair_keys]
    return r_vals, curves[idx_a], curves[idx_b], pair_meta


def run_set_vs_reference(
    df: pd.DataFrame,
    *,
    reference_set: str = REFERENCE_SET,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    r_ranges: tuple[tuple[str, float, float], ...] = DEFAULT_R_RANGES,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    extra_id_cols: tuple[str, ...] = (),
    label_col: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    sets = sorted(s for s in df["set_name"].dropna().unique() if s != reference_set)
    ref_df = df[df["set_name"] == reference_set]
    if ref_df.empty:
        raise ValueError(f"Reference set {reference_set!r} not found in data")

    for metric in metrics:
        r_vals_ref, curves_ref, _ = extract_curves_matrix(
            ref_df, metric, label_col=label_col, extra_id_cols=extra_id_cols
        )
        for set_name in sets:
            other = df[df["set_name"] == set_name]
            r_vals_o, curves_o, _ = extract_curves_matrix(
                other, metric, label_col=label_col, extra_id_cols=extra_id_cols
            )
            if curves_o.shape[0] == 0 or curves_ref.shape[0] == 0:
                continue
            # Align r grids (should already match; reindex defensively)
            if not np.array_equal(r_vals_ref, r_vals_o):
                r_vals = np.union1d(r_vals_ref, r_vals_o)
                curves_ref_u = _reindex_curves(curves_ref, r_vals_ref, r_vals)
                curves_o_u = _reindex_curves(curves_o, r_vals_o, r_vals)
            else:
                r_vals = r_vals_ref
                curves_ref_u = curves_ref
                curves_o_u = curves_o

            for range_label, r_min, r_max in r_ranges:
                # Independent RNG stream per comparison so order of loops doesn't matter
                rng = np.random.default_rng(
                    _seed_for(seed, "unpaired", metric, set_name, range_label)
                )
                result_rows = unpaired_permutation_test(
                    curves_o_u,  # A = comparison set
                    curves_ref_u,  # B = reference (15F1)
                    r_vals,
                    r_min=r_min,
                    r_max=r_max,
                    n_perm=n_perm,
                    rng=rng,
                )
                for result in result_rows:
                    rows.append(
                        {
                            "comparison": f"{set_name}_vs_{reference_set}",
                            "group_a": set_name,
                            "group_b": reference_set,
                            "metric": metric,
                            "r_range": range_label,
                            "r_min_nm": r_min,
                            "r_max_nm": r_max,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def run_monomer_vs_dimer(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    r_ranges: tuple[tuple[str, float, float], ...] = DEFAULT_R_RANGES,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    label_col: str = "aunp_pick_label",
    extra_id_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        r_vals, curves_m, curves_d, _ = align_paired_curves(
            df,
            metric,
            label_col=label_col,
            label_a="monomer",
            label_b="dimer",
            extra_id_cols=extra_id_cols,
        )
        if curves_m.shape[0] == 0:
            continue
        for range_label, r_min, r_max in r_ranges:
            rng = np.random.default_rng(
                _seed_for(seed, "paired", metric, "monomer_dimer", range_label)
            )
            result_rows = paired_permutation_test(
                curves_m,
                curves_d,
                r_vals,
                r_min=r_min,
                r_max=r_max,
                n_perm=n_perm,
                rng=rng,
            )
            for result in result_rows:
                rows.append(
                    {
                        "comparison": "monomer_vs_dimer",
                        "group_a": "monomer",
                        "group_b": "dimer",
                        "metric": metric,
                        "r_range": range_label,
                        "r_min_nm": r_min,
                        "r_max_nm": r_max,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _reindex_curves(curves: np.ndarray, r_old: np.ndarray, r_new: np.ndarray) -> np.ndarray:
    out = np.full((curves.shape[0], len(r_new)), np.nan, dtype=float)
    old_index = {round(float(r), 6): i for i, r in enumerate(r_old)}
    for j, r in enumerate(r_new):
        i = old_index.get(round(float(r), 6))
        if i is not None:
            out[:, j] = curves[:, i]
    return out


def _seed_for(base: int, *parts: str) -> int:
    """Deterministic child seed from base + string parts."""
    payload = f"{base}|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _prepare_dataset_frame(
    df: pd.DataFrame,
    *,
    curve_type_filter: str | None,
    label_col: str,
    metrics: tuple[str, ...] = (),
    long_family_format: bool = False,
) -> pd.DataFrame:
    """Filter to observed curves and normalize metric columns when needed."""
    out = df.copy()
    if curve_type_filter is not None:
        if "curve_type" not in out.columns:
            raise ValueError("curve_type_filter set but no curve_type column in data")
        out = out[out["curve_type"] == curve_type_filter].copy()
    # Fusion monomer/dimer uses aunp_subset; drop pooled 'both' from pairing table source.
    if label_col == "aunp_subset" and "aunp_subset" in out.columns:
        out = out[out["aunp_subset"].isin(["monomer", "dimer", "all"])].copy()

    # Bidirectional long format: family + value → wide columns named by family.
    if long_family_format:
        if "family" not in out.columns or "value" not in out.columns:
            raise ValueError("long_family_format requires 'family' and 'value' columns")
        if metrics:
            out = out[out["family"].isin(metrics)].copy()
        id_cols = [
            c
            for c in (
                "tomogram_name",
                "alignment_dir",
                "cleft_name",
                "aunp_subset",
                "window_mode",
                "curve_type",
                "replicate_index",
                "r_nm",
                "n_aunp_partners",
                "n_query_points",
                "window_volume_nm3",
                "set_name",
            )
            if c in out.columns
        ]
        # Keep optional label columns only when they have any non-null values (all-NaN
        # index keys make pivot_table drop every row).
        for opt in ("aunp_pick_label",):
            if opt in out.columns and out[opt].notna().any():
                id_cols.append(opt)
        # One value per id × family; pivot families to columns.
        wide = out.pivot_table(
            index=id_cols,
            columns="family",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        out = wide

    return out


def _load_csv_filtered(
    path: Path,
    *,
    curve_type_filter: str | None,
    label_col: str,
    metrics: tuple[str, ...],
    long_family_format: bool,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load CSV, optionally chunk-filtering large bidirectional exports before pivoting."""
    if not long_family_format:
        return _prepare_dataset_frame(
            pd.read_csv(path),
            curve_type_filter=curve_type_filter,
            label_col=label_col,
            metrics=metrics,
            long_family_format=False,
        )

    # Chunked read for multi-GB bidirectional files.
    keep_cols = [
        "tomogram_name",
        "alignment_dir",
        "cleft_name",
        "aunp_subset",
        "window_mode",
        "curve_type",
        "family",
        "replicate_index",
        "r_nm",
        "value",
        "n_aunp_partners",
        "n_query_points",
        "window_volume_nm3",
        "set_name",
        "aunp_pick_label",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in keep_cols, chunksize=chunksize):
        m = pd.Series(True, index=chunk.index)
        if curve_type_filter is not None:
            m &= chunk["curve_type"] == curve_type_filter
        if metrics:
            m &= chunk["family"].isin(metrics)
        if label_col == "aunp_subset" and "aunp_subset" in chunk.columns:
            m &= chunk["aunp_subset"].isin(["monomer", "dimer", "all"])
        sub = chunk.loc[m]
        if len(sub):
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return _prepare_dataset_frame(
        pd.concat(frames, ignore_index=True),
        curve_type_filter=None,  # already filtered
        label_col=label_col,
        metrics=metrics,
        long_family_format=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        default="cleft_center",
        help="Which Ripley's_stats_curves_final_data subfolder to analyze",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override input directory (default: data/Ripley's_stats_curves_final_data/<dataset>)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <data-dir>/permutation_tests)",
    )
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reference-set", type=str, default=REFERENCE_SET)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    data_dir = args.data_dir or Path("data/Ripley's_stats_curves_final_data") / args.dataset
    out_dir = args.out_dir or (data_dir / "permutation_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg["out_prefix"]
    metrics = tuple(cfg["metrics"])
    label_col = cfg["label_col"]
    extra_id_cols = tuple(cfg["extra_id_cols"])
    curve_type_filter = cfg["curve_type_filter"]
    long_family_format = bool(cfg.get("long_family_format", False))

    sets_csv = data_dir / cfg["sets_csv"]
    md_csv = data_dir / cfg["monomer_dimer_csv"]

    print(f"Dataset: {args.dataset}")
    print(f"Loading {sets_csv}")
    sets_df = _load_csv_filtered(
        sets_csv,
        curve_type_filter=curve_type_filter,
        label_col=label_col,
        metrics=metrics,
        long_family_format=long_family_format,
    )
    print(
        f"  {len(sets_df):,} rows; sets: {sorted(sets_df['set_name'].dropna().unique())}; "
        f"metrics={metrics}"
    )

    print(f"Loading {md_csv}")
    md_df = _load_csv_filtered(
        md_csv,
        curve_type_filter=curve_type_filter,
        label_col=label_col,
        metrics=metrics,
        long_family_format=long_family_format,
    )
    if label_col in md_df.columns:
        labels = sorted(md_df[label_col].dropna().astype(str).unique())
    else:
        labels = []
    print(f"  {len(md_df):,} rows; {label_col}: {labels}")

    print(
        f"\nRunning set-vs-{args.reference_set} unpaired permutation tests "
        f"(n_perm={args.n_perm})..."
    )
    set_results = run_set_vs_reference(
        sets_df,
        reference_set=args.reference_set,
        metrics=metrics,
        n_perm=args.n_perm,
        seed=args.seed,
        extra_id_cols=extra_id_cols,
        label_col=label_col,
    )
    set_out = out_dir / f"{prefix}_set_vs_15F1_curve_permutation.csv"
    set_results.to_csv(set_out, index=False)
    print(f"  Wrote {set_out} ({len(set_results)} rows)")

    print(f"\nRunning monomer-vs-dimer paired permutation tests (n_perm={args.n_perm})...")
    md_results = run_monomer_vs_dimer(
        md_df,
        metrics=metrics,
        n_perm=args.n_perm,
        seed=args.seed,
        label_col=label_col,
        extra_id_cols=extra_id_cols,
    )
    md_out = out_dir / f"{prefix}_monomer_vs_dimer_curve_permutation.csv"
    md_results.to_csv(md_out, index=False)
    print(f"  Wrote {md_out} ({len(md_results)} rows)")

    cols = [
        "comparison",
        "metric",
        "r_range",
        "statistic",
        "n_a",
        "n_b",
        "t_obs_signed",
        "t_obs",
        "p_value",
    ]
    print("\n=== Set vs 15F1 (unpaired) ===")
    if set_results.empty:
        print("  (no comparisons)")
    else:
        print(set_results[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Monomer vs dimer (paired) ===")
    if md_results.empty:
        print("  (no paired units found)")
    else:
        print(md_results[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()