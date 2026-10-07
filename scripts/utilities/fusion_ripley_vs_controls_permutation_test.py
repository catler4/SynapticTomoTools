#!/usr/bin/env python3
"""
Fusion-site Ripley L_combined: observed fusing vs controls (pooled across label sets).

Two designs:

1. Fusing vs adjacent (close) — paired permutation (sign-flip) on zones that have both
   a fusing and a close curve. T = ∫ [δ(r)]² dr where δ = mean(fusing − close).

2. Fusing vs 40 nm shift / label permutation — Monte Carlo test against per-zone null
   replicate curves (100 replicates). Pool the observed fusing mean across zones; for
   each replicate id, pool that replicate’s null curves across the same zones. Diggle-
   style reference mean μ = (M_obs + Σ M_null) / (N + 1). Then
   T = ∫(curve − μ)² dr for the observed and each null curve.
   p = (1 + #{T_null ≥ T_obs}) / (1 + N).

Datasets:
- fusion_point_aunp_ripley_bidirectional_curves.csv — pool all set_names
- fusion_point_aunp_ripley_bidirectional_curves_monomer_dimer.csv — monomer and dimer
  separately (aunp_subset)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ripley_curve_group_permutation_test import (  # noqa: E402
    DEFAULT_N_PERM,
    DEFAULT_SEED,
    STAT_INTEGRATED_SQ_DIFF,
    _mc_p,
    _r_mask,
    _seed_for,
    extract_curves_matrix,
    paired_permutation_test,
)

# Fusion-vs-controls radius windows (nm).
FUSION_R_RANGES = (
    ("0-100", 0.0, 100.0),
    ("0-200", 0.0, 200.0),
    ("0-500", 0.0, 500.0),
)

DATA_DIR = Path("data/Ripley's_stats_curves_final_data/fusion_sites")
OUT_DIR = DATA_DIR / "permutation_tests"
SETS_CSV = "fusion_point_aunp_ripley_bidirectional_curves.csv"
MD_CSV = "fusion_point_aunp_ripley_bidirectional_curves_monomer_dimer.csv"
METRIC = "l_combined"
ZONE_KEYS = ("tomogram_name", "alignment_dir", "cleft_name")
CONTROL_CLOSE = "close"
CONTROL_SHIFT = "shift_40nm"
CONTROL_PERM = "label_permutation"
CURVE_TYPES_KEEP = ("fusing", CONTROL_CLOSE, CONTROL_SHIFT, CONTROL_PERM)


def load_bidirectional_l_combined(
    path: Path,
    *,
    aunp_subsets: tuple[str, ...] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Chunk-load bidirectional CSV → wide table with l_combined column (all curve types)."""
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
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in keep_cols, chunksize=chunksize):
        m = (chunk["family"] == METRIC) & (chunk["curve_type"].isin(CURVE_TYPES_KEEP))
        if aunp_subsets is not None:
            m &= chunk["aunp_subset"].isin(aunp_subsets)
        sub = chunk.loc[m]
        if len(sub):
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
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
        if c in long.columns
    ]
    wide = (
        long.pivot_table(index=id_cols, columns="family", values="value", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None
    return wide


def curves_for_type(
    df: pd.DataFrame,
    curve_type: str,
    *,
    extra_id_cols: tuple[str, ...] = ("replicate_index",),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sub = df[df["curve_type"] == curve_type]
    return extract_curves_matrix(
        sub, METRIC, label_col=None, extra_id_cols=extra_id_cols
    )


def _zone_key_frame(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[list(ZONE_KEYS)].copy()


def align_zone_pairs(
    r_vals: np.ndarray,
    curves_a: np.ndarray,
    meta_a: pd.DataFrame,
    curves_b: np.ndarray,
    meta_b: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Inner-join curves on zone keys (tomogram × alignment × cleft)."""
    a = meta_a.copy()
    b = meta_b.copy()
    a["_ia"] = np.arange(len(a))
    b["_ib"] = np.arange(len(b))
    merged = a[list(ZONE_KEYS) + ["_ia"]].merge(
        b[list(ZONE_KEYS) + ["_ib"]], on=list(ZONE_KEYS), how="inner"
    )
    if merged.empty:
        return r_vals, np.empty((0, len(r_vals))), np.empty((0, len(r_vals))), pd.DataFrame()
    ia = merged["_ia"].to_numpy(dtype=int)
    ib = merged["_ib"].to_numpy(dtype=int)
    return r_vals, curves_a[ia], curves_b[ib], merged[list(ZONE_KEYS)]


def pool_null_replicates_by_id(
    df: pd.DataFrame,
    curve_type: str,
    zone_keys: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For zones in zone_keys, build (n_replicates, n_r) matrix of pooled mean null curves.

    Each row is the equal-weight mean across zones of that replicate_index.
    """
    sub = df[df["curve_type"] == curve_type].copy()
    # Restrict to requested zones
    sub = sub.merge(zone_keys.drop_duplicates(), on=list(ZONE_KEYS), how="inner")
    if sub.empty:
        return np.array([]), np.empty((0, 0))

    r_vals, curves, meta = extract_curves_matrix(
        sub, METRIC, label_col=None, extra_id_cols=("replicate_index",)
    )
    if curves.shape[0] == 0:
        return r_vals, np.empty((0, 0))

    reps = sorted(meta["replicate_index"].dropna().unique().tolist())
    pooled = []
    for rep in reps:
        mask = (meta["replicate_index"] == rep).to_numpy()
        if not np.any(mask):
            continue
        pooled.append(np.nanmean(curves[mask], axis=0))
    if not pooled:
        return r_vals, np.empty((0, len(r_vals)))
    return r_vals, np.vstack(pooled)


def diggle_integrated_null_test(
    observed_mean: np.ndarray,
    null_means: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_min: float,
    r_max: float,
) -> list[dict]:
    """
    Diggle-pooled μ = (obs + Σ null) / (N+1); T = ∫(curve−μ)² on the radius window.
    """
    observed_mean = np.asarray(observed_mean, dtype=float).reshape(-1)
    null_means = np.atleast_2d(np.asarray(null_means, dtype=float))
    n_null = int(null_means.shape[0])
    mask = _r_mask(r_vals, r_min, r_max)
    if n_null < 1 or not np.any(mask) or len(observed_mean) != len(r_vals):
        return [
            {
                "statistic": STAT_INTEGRATED_SQ_DIFF,
                "t_obs_signed": np.nan,
                "t_obs": np.nan,
                "p_value": np.nan,
                "n_perm": n_null,
                "n_perm_finite": 0,
                "note": "insufficient nulls",
            }
        ]

    r = r_vals[mask]
    obs = observed_mean[mask]
    nulls = null_means[:, mask]
    all_curves = np.vstack([obs[None, :], nulls])
    mu = np.nanmean(all_curves, axis=0)

    def _sq(curve: np.ndarray) -> float:
        d = curve - mu
        ok = np.isfinite(d) & np.isfinite(r)
        if ok.sum() < 2:
            return float("nan")
        return float(np.trapezoid(d[ok] ** 2, r[ok]))

    t_sq = _sq(obs)
    null_sq = np.array([_sq(nulls[i]) for i in range(n_null)], dtype=float)
    p_sq, n_ok_sq = _mc_p(t_sq, null_sq)
    return [
        {
            "statistic": STAT_INTEGRATED_SQ_DIFF,
            "t_obs_signed": np.nan,
            "t_obs": t_sq,
            "p_value": p_sq,
            "n_perm": n_null,
            "n_perm_finite": n_ok_sq,
            "note": "Diggle μ; t_obs = ∫(M_obs − μ)² dr",
        }
    ]


def vesicle_summary(meta: pd.DataFrame) -> dict:
    """n zones and sum of n_query_points (vesicles) from curve meta rows."""
    if meta.empty or "n_query_points" not in meta.columns:
        return {"n_zones": int(len(meta)), "n_vesicles": np.nan}
    return {
        "n_zones": int(len(meta)),
        "n_vesicles": int(meta["n_query_points"].sum()),
    }


def run_panel(
    df: pd.DataFrame,
    *,
    panel_name: str,
    n_perm_close: int,
    seed: int,
) -> pd.DataFrame:
    """Run fusing vs close / shift / label_permutation for one panel (all / monomer / dimer)."""
    rows: list[dict] = []

    r_f, curves_f, meta_f = curves_for_type(df, "fusing")
    if curves_f.shape[0] == 0:
        return pd.DataFrame()
    fusing_mean = np.nanmean(curves_f, axis=0)
    f_sum = vesicle_summary(meta_f)

    # --- close (paired) ---
    r_c, curves_c, meta_c = curves_for_type(df, CONTROL_CLOSE)
    r_p, cur_f_p, cur_c_p, pair_meta = align_zone_pairs(
        r_f, curves_f, meta_f, curves_c, meta_c
    )
    # vesicle counts on paired zones only
    if not pair_meta.empty:
        paired_f = meta_f.merge(pair_meta, on=list(ZONE_KEYS), how="inner")
        paired_c = meta_c.merge(pair_meta, on=list(ZONE_KEYS), how="inner")
        close_f_sum = vesicle_summary(paired_f)
        close_c_sum = vesicle_summary(paired_c)
    else:
        close_f_sum = {"n_zones": 0, "n_vesicles": 0}
        close_c_sum = {"n_zones": 0, "n_vesicles": 0}

    for range_label, r_min, r_max in FUSION_R_RANGES:
        rng = np.random.default_rng(
            _seed_for(seed, panel_name, "close", range_label)
        )
        if cur_f_p.shape[0] == 0:
            continue
        for result in paired_permutation_test(
            cur_f_p,
            cur_c_p,
            r_p,
            r_min=r_min,
            r_max=r_max,
            n_perm=n_perm_close,
            rng=rng,
        ):
            rows.append(
                {
                    "panel": panel_name,
                    "comparison": "fusing_vs_close",
                    "group_a": "fusing",
                    "group_b": "close",
                    "metric": METRIC,
                    "r_range": range_label,
                    "r_min_nm": r_min,
                    "r_max_nm": r_max,
                    "design": "paired_sign_flip",
                    "n_a": close_f_sum["n_zones"],
                    "n_b": close_c_sum["n_zones"],
                    "n_pairs": int(cur_f_p.shape[0]),
                    "n_vesicles_a": close_f_sum["n_vesicles"],
                    "n_vesicles_b": close_c_sum["n_vesicles"],
                    **result,
                }
            )

    # --- shift / label permutation (MC Diggle on pooled means) ---
    zone_keys = _zone_key_frame(meta_f)
    for control, control_label in (
        (CONTROL_SHIFT, "fusing_vs_shift_40nm"),
        (CONTROL_PERM, "fusing_vs_label_permutation"),
    ):
        r_n, null_means = pool_null_replicates_by_id(df, control, zone_keys)
        # Align r if needed (should match)
        if len(r_n) and not np.array_equal(r_n, r_f):
            # reindex null means onto r_f
            aligned = np.full((null_means.shape[0], len(r_f)), np.nan)
            idx = {round(float(r), 6): i for i, r in enumerate(r_n)}
            for j, r in enumerate(r_f):
                i = idx.get(round(float(r), 6))
                if i is not None:
                    aligned[:, j] = null_means[:, i]
            null_means = aligned
            r_use = r_f
        else:
            r_use = r_f if len(r_f) else r_n

        # n_vesicles for null: use one replicate's meta sum as representative
        null_meta_rep0 = df[
            (df["curve_type"] == control) & (df["replicate_index"] == 0)
        ]
        if not null_meta_rep0.empty:
            _, _, meta_n0 = extract_curves_matrix(
                null_meta_rep0.merge(zone_keys, on=list(ZONE_KEYS), how="inner"),
                METRIC,
                extra_id_cols=("replicate_index",),
            )
            # Prefer zone-matched fusing vesicle count for "a"; null query count for b
            n_sum = vesicle_summary(meta_n0) if len(meta_n0) else {"n_zones": 0, "n_vesicles": 0}
        else:
            n_sum = {"n_zones": 0, "n_vesicles": 0}

        for range_label, r_min, r_max in FUSION_R_RANGES:
            for result in diggle_integrated_null_test(
                fusing_mean, null_means, r_use, r_min=r_min, r_max=r_max
            ):
                rows.append(
                    {
                        "panel": panel_name,
                        "comparison": control_label,
                        "group_a": "fusing",
                        "group_b": control,
                        "metric": METRIC,
                        "r_range": range_label,
                        "r_min_nm": r_min,
                        "r_max_nm": r_max,
                        "design": "diggle_mc_null_pooled_means",
                        "n_a": f_sum["n_zones"],
                        "n_b": int(null_means.shape[0]),  # n null replicates
                        "n_pairs": f_sum["n_zones"],
                        "n_vesicles_a": f_sum["n_vesicles"],
                        "n_vesicles_b": n_sum["n_vesicles"],
                        **result,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-perm-close", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets_path = args.data_dir / SETS_CSV
    md_path = args.data_dir / MD_CSV

    print(f"Loading pooled sets from {sets_path}")
    sets_df = load_bidirectional_l_combined(sets_path, aunp_subsets=("all",))
    print(
        f"  {len(sets_df):,} rows; curve_types={sorted(sets_df['curve_type'].unique())}; "
        f"sets={sorted(sets_df['set_name'].dropna().unique())}"
    )

    print(f"Loading monomer/dimer from {md_path}")
    md_df = load_bidirectional_l_combined(md_path, aunp_subsets=("monomer", "dimer"))
    print(
        f"  {len(md_df):,} rows; subsets={sorted(md_df['aunp_subset'].unique())}; "
        f"curve_types={sorted(md_df['curve_type'].unique())}"
    )

    all_rows: list[pd.DataFrame] = []

    print("\n=== Panel: all sets pooled (aunp_subset=all) ===")
    panel_all = run_panel(
        sets_df, panel_name="all_sets_pooled", n_perm_close=args.n_perm_close, seed=args.seed
    )
    all_rows.append(panel_all)
    print(panel_all[["comparison", "r_range", "statistic", "n_a", "n_b", "n_vesicles_a", "p_value"]].to_string(index=False))

    for subset in ("monomer", "dimer"):
        print(f"\n=== Panel: {subset} ===")
        sub_df = md_df[md_df["aunp_subset"] == subset].copy()
        panel = run_panel(
            sub_df,
            panel_name=subset,
            n_perm_close=args.n_perm_close,
            seed=args.seed,
        )
        all_rows.append(panel)
        if panel.empty:
            print("  (empty)")
        else:
            print(
                panel[
                    ["comparison", "r_range", "statistic", "n_a", "n_b", "n_vesicles_a", "p_value"]
                ].to_string(index=False)
            )

    out = pd.concat([r for r in all_rows if not r.empty], ignore_index=True)
    out_path = args.out_dir / "fusion_sites_fusing_vs_controls_l_combined_permutation.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
