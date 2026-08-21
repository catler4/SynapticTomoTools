#!/usr/bin/env python3
"""
Streamlit-based Active Zonogram Viewer

Quickly browse active zonogram images (position + main) across tomogram groups,
tomograms, and alignment directories (best_alignment, liza_az0, liza_az1, etc.).

Usage:
    streamlit run scripts/utilities/zonogram_viewer.py
    streamlit run scripts/utilities/zonogram_viewer.py --server.address 0.0.0.0 --server.port 8501

Or with custom data path:
    streamlit run scripts/utilities/zonogram_viewer.py -- --data-dir /path/to/data

In the sidebar, optional **Tomogram CSV** (columns ``tomoname``, ``set``, ``alignment_dir``) limits Prev/Next
and the tomogram list to those rows only; each row’s ``alignment_dir`` is the only alignment scanned
for zonograms (same layout as pipeline CSVs under ``tomogram_csv_files/``). With a CSV loaded, **Group (from CSV)**
filters rows by the ``set`` column (choose **All** for every row in the file).
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import mrcfile
import numpy as np
import streamlit as st
from PIL import Image

ASSIGNMENTS_CSV = "zonogram_viewer_assignments.csv"
PROPERTY_OPTIONS = ["", "include", "improve", "exclude"]
SAMPLE_TYPE_OPTIONS = ["", "tissue", "semi-tissue", "synaptosome"]
AUNP_PICK_QUALITY_OPTIONS = ["", "good", "a few missing", "many missing/wrong"]
MEMBRANE_SEG_QUALITY_OPTIONS = ["", "complete", "small issues", "significant issues"]
REDEPOSITION_ISSUES_OPTIONS = ["", "yes", "no"]

# Add project root to path (scripts/utilities/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Possible locations for active_zonograms under an alignment dir
AZ_SUBDIRS = [
    "active_zonograms",
    "STT_results/visualizations/active_zonograms",
]


def find_az_dirs(alignment_path: Path) -> list[Path]:
    """Find directories containing active zonogram images under an alignment dir."""
    found = []
    for sub in AZ_SUBDIRS:
        az_dir = alignment_path / sub
        if az_dir.exists():
            # Check for at least one zonogram image
            if list(az_dir.glob("active_zonogram_*_position.png")) or list(az_dir.glob("active_zonogram_*.png")):
                found.append(az_dir)
    return found


def get_az_pairs(az_dir: Path) -> list[tuple[int, Path | None, Path | None]]:
    """
    Return list of (az_idx, position_path, main_path) for each active zone.
    Main: active_zonogram_N.png, or active_zonogram_N_selected_aunps.png, or similar.
    """
    pos_imgs = sorted(az_dir.glob("active_zonogram_*_position.png"))
    pairs = []
    main_candidates = ("active_zonogram_{idx}.png", "active_zonogram_{idx}_selected_aunps.png")
    for pos in pos_imgs:
        m = re.search(r"active_zonogram_(\d+)_position\.png", pos.name)
        if not m:
            continue
        idx = int(m.group(1))
        main = None
        for tmpl in main_candidates:
            p = az_dir / tmpl.format(idx=idx)
            if p.exists():
                main = p
                break
        pairs.append((idx, pos, main))
    # If no position images, try main-only
    if not pairs:
        for main in sorted(az_dir.glob("active_zonogram_*.png")):
            m = re.search(r"active_zonogram_(\d+)\.png", main.name)
            if not m or "_position" in main.name:
                continue
            idx = int(m.group(1))
            pos = az_dir / f"active_zonogram_{idx}_position.png"
            pairs.append((idx, pos if pos.exists() else None, main))
    return sorted(pairs, key=lambda x: x[0])


def discover_tomograms(data_root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Discover tomogram groups from directory structure: data_root/{group}/TOP_TOMOS/{tomo}
    Groups = subdirectories of data_root (e.g. 15F1, 15F1-H4K2Cys, 15F1and5F11dimer).
    Returns {group_name: [(tomoname, set_name), ...]} - set_name equals group_name.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    data_root = Path(data_root)

    for set_dir in sorted(data_root.iterdir()):
        if not set_dir.is_dir():
            continue
        top_tomos = set_dir / "TOP_TOMOS"
        if top_tomos.exists():
            tomos = [(d.name, set_dir.name) for d in top_tomos.iterdir() if d.is_dir()]
            if tomos:
                groups[set_dir.name] = sorted(tomos, key=lambda x: x[0])

    return groups


def get_tomogram_path(data_root: Path, set_name: str, tomo_name: str) -> Path | None:
    """Get full path to tomogram directory."""
    candidates = [
        data_root / set_name / "TOP_TOMOS" / tomo_name,
        data_root / tomo_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_assignments(csv_path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Load assignments from CSV. Key = (group, tomoname, alignment_dir, active_zone)."""
    out: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = row.get("group", "").strip()
            t = row.get("tomoname", "").strip()
            al = row.get("alignment_directory", "").strip()
            az = str(row.get("active_zone", "")).strip()
            if g and t:
                out[(g, t, al, az)] = {
                    "property": row.get("property", "").strip(),
                    "sample_type": row.get("sample_type", "").strip(),
                    "aunp_pick_quality": row.get("aunp_pick_quality", "").strip(),
                    "membrane_segmentation_quality": row.get("membrane_segmentation_quality", "").strip(),
                    "redeposition_issues": row.get("redeposition_issues", "").strip(),
                }
    return out


def save_assignment(
    csv_path: Path,
    group: str,
    tomoname: str,
    alignment_directory: str,
    active_zone: str,
    property_value: str,
    sample_type_value: str,
    aunp_pick_quality_value: str,
    membrane_segmentation_quality_value: str,
    redeposition_issues_value: str,
) -> None:
    """Update one assignment and write CSV. Preserves other rows and their timestamps."""
    key = (group, tomoname, alignment_directory, active_zone)
    rows_dict: dict[tuple[str, str, str, str], dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                g = row.get("group", "").strip()
                t = row.get("tomoname", "").strip()
                al = row.get("alignment_directory", "").strip()
                az = str(row.get("active_zone", "")).strip()
                if g and t:
                    rkey = (g, t, al, az)
                    rows_dict[rkey] = {
                        "group": g,
                        "tomoname": t,
                        "alignment_directory": al,
                        "active_zone": az,
                        "property": row.get("property", "").strip(),
                        "sample_type": row.get("sample_type", "").strip(),
                        "aunp_pick_quality": row.get("aunp_pick_quality", "").strip(),
                        "membrane_segmentation_quality": row.get("membrane_segmentation_quality", "").strip(),
                        "redeposition_issues": row.get("redeposition_issues", "").strip(),
                        "updated_at": row.get("updated_at", ""),
                    }
    now = datetime.now().isoformat()
    rows_dict[key] = {
        "group": group,
        "tomoname": tomoname,
        "alignment_directory": alignment_directory,
        "active_zone": active_zone,
        "property": property_value,
        "sample_type": sample_type_value,
        "aunp_pick_quality": aunp_pick_quality_value,
        "membrane_segmentation_quality": membrane_segmentation_quality_value,
        "redeposition_issues": redeposition_issues_value,
        "updated_at": now,
    }
    rows = sorted(rows_dict.values(), key=lambda r: (r["group"], r["tomoname"], r["alignment_directory"], r["active_zone"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "tomoname",
                "alignment_directory",
                "active_zone",
                "property",
                "sample_type",
                "aunp_pick_quality",
                "membrane_segmentation_quality",
                "redeposition_issues",
                "updated_at",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def get_alignment_dirs(tomo_path: Path) -> list[tuple[str, Path]]:
    """
    Get (name, az_dir) for each alignment that has active zonograms.
    e.g. best_alignment, liza_az0, liza_az1
    """
    results = []
    for sub in sorted(tomo_path.iterdir()):
        if not sub.is_dir():
            continue
        az_dirs = find_az_dirs(sub)
        for az_dir in az_dirs:
            results.append((sub.name, az_dir))
    return results


def get_alignment_dirs_named(tomo_path: Path, alignment_name: str) -> list[tuple[str, Path]]:
    """Return zonogram dirs only under ``tomo_path / alignment_name`` (for CSV-filtered browsing)."""
    sub = tomo_path / alignment_name.strip()
    if not sub.is_dir():
        return []
    results: list[tuple[str, Path]] = []
    for az_dir in find_az_dirs(sub):
        results.append((alignment_name.strip(), az_dir))
    return results


def parse_tomogram_csv(csv_path: Path) -> tuple[list[tuple[str, str, str]], str | None]:
    """
    Load pipeline-style tomogram CSV.
    Returns (list of (tomoname, set_name, alignment_dir), error_message_or_None).
    """
    required = ("tomoname", "set", "alignment_dir")
    rows: list[tuple[str, str, str]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], "CSV has no header row."
            fields_lower = {h.strip().lower(): h for h in reader.fieldnames if h}
            missing = [c for c in required if c not in fields_lower]
            if missing:
                return [], f"Missing columns {missing}. Need: {', '.join(required)}."
            h_t = fields_lower["tomoname"]
            h_s = fields_lower["set"]
            h_a = fields_lower["alignment_dir"]
            for row in reader:
                t = (row.get(h_t) or "").strip()
                s = (row.get(h_s) or "").strip()
                a = (row.get(h_a) or "").strip()
                if not t:
                    continue
                if not s or not a or a.lower() == "nan":
                    return [], f"Row with tomoname '{t}' missing set or alignment_dir."
                rows.append((t, s, a))
    except OSError as e:
        return [], str(e)
    if not rows:
        return [], "No data rows in CSV."
    return rows, None


def infer_alignment_dir_from_az_dir(az_dir: Path, tomo_path: Path) -> Path | None:
    """Infer alignment directory from an active_zonograms directory path."""
    for candidate in [az_dir, *az_dir.parents]:
        if candidate.parent == tomo_path:
            return candidate
    return None


def get_ddw_mrc_path(alignment_dir: Path) -> Path | None:
    """Find a *ddw.mrc tomogram file in an alignment directory."""
    ddw_files = sorted(alignment_dir.glob("*ddw.mrc"))
    if not ddw_files:
        return None
    # Prefer full reconstruction naming if present.
    for p in ddw_files:
        if "full_rec" in p.name:
            return p
    return ddw_files[0]


def get_slice_index(nz: int, slice_mode: str) -> int:
    if slice_mode == "-2":
        return int(2.0 / 8.0 * (nz - 1))
    if slice_mode == "-1":
        return int(3.0 / 8.0 * (nz - 1))
    if slice_mode == "central":
        return int(4.0 / 8.0 * (nz - 1))
    if slice_mode == "+1":
        return int(5.0 / 8.0 * (nz - 1))
    if slice_mode == "+2":
        return int(6.0 / 8.0 * (nz - 1))
    return nz // 2


def get_slice_suffix(slice_mode: str) -> str:
    if slice_mode == "-2":
        return "_q1_slice.png"
    if slice_mode == "-1":
        return "_eighth3_slice.png"
    if slice_mode == "central":
        return "_central_slice.png"
    if slice_mode == "+1":
        return "_eighth5_slice.png"
    if slice_mode == "+2":
        return "_q3_slice.png"
    return "_central_slice.png"


def ensure_tomogram_slice_png(ddw_mrc_path: Path, slice_mode: str) -> Path:
    """
    Return cached PNG path for selected tomogram slice.
    Generates (or regenerates) PNG if needed.
    """
    snapshots_dir = ddw_mrc_path.parent / "ddw_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    png_path = snapshots_dir / f"{ddw_mrc_path.stem}{get_slice_suffix(slice_mode)}"
    if png_path.exists() and png_path.stat().st_mtime >= ddw_mrc_path.stat().st_mtime:
        return png_path

    with mrcfile.open(ddw_mrc_path, permissive=True) as mrc:
        vol = np.asarray(mrc.data)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D MRC volume, got shape {vol.shape}")

    z_idx = get_slice_index(vol.shape[0], slice_mode)
    sl = vol[z_idx].astype(np.float32)

    # Robust normalization for display.
    lo, hi = np.percentile(sl, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(sl)), float(np.max(sl))
    if hi <= lo:
        norm = np.zeros_like(sl, dtype=np.uint8)
    else:
        clipped = np.clip(sl, lo, hi)
        norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    Image.fromarray(norm, mode="L").save(png_path)
    return png_path


def main():
    st.set_page_config(page_title="Active Zonogram Viewer", layout="wide")
    st.title("Active Zonogram Viewer")
    st.caption("Browse position + main zonogram images by group, tomogram, and alignment directory.")

    # Sidebar: data root
    default_data = "/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms"
    data_root = st.sidebar.text_input(
        "Data root directory",
        value=default_data,
        help="Root containing set/TOP_TOMOS/tomogram structure",
    )
    data_path = Path(data_root)
    if not data_path.exists():
        st.sidebar.error(f"Path does not exist: {data_path}")
        st.info("Set the data root in the sidebar to the directory containing your tomogram sets.")
        return

    csv_input = st.sidebar.text_input(
        "Tomogram CSV (optional)",
        value="",
        help=(
            "Path to a CSV with columns tomoname, set, alignment_dir (same as pipeline). "
            "Relative paths are resolved from the SynapticTomoTools repo root. "
            "When set, only those tomograms and alignment directories are browsed."
        ),
    )
    csv_path_raw = csv_input.strip()
    csv_rows_all: list[tuple[str, str, str]] | None = None
    csv_parse_error: str | None = None
    csv_p: Path | None = None
    if csv_path_raw:
        csv_p = Path(csv_path_raw)
        if not csv_p.is_absolute():
            csv_p = (PROJECT_ROOT / csv_p).resolve()
        else:
            csv_p = csv_p.resolve()
        if not csv_p.exists():
            csv_parse_error = f"CSV not found: {csv_p}"
        else:
            csv_rows_all, csv_parse_error = parse_tomogram_csv(csv_p)
            if csv_parse_error:
                csv_rows_all = None
    if csv_parse_error:
        st.sidebar.warning(csv_parse_error)

    csv_mode = bool(csv_rows_all)
    csv_rows: list[tuple[str, str, str]] | None = None
    csv_group_pick = "All"
    if csv_mode:
        assert csv_rows_all is not None
        assert csv_p is not None
        unique_sets = sorted({r[1] for r in csv_rows_all})
        csv_group_options = ["All"] + unique_sets
        csv_group_pick = st.sidebar.selectbox(
            "Group (from CSV)",
            csv_group_options,
            index=0,
            help="Restrict the tomogram list to one ``set`` value from the CSV, or All for every row.",
            key=f"zonogram_csv_group_{csv_p}",
        )
        if csv_group_pick == "All":
            csv_rows = list(csv_rows_all)
        else:
            csv_rows = [r for r in csv_rows_all if r[1] == csv_group_pick]
        if not csv_rows:
            st.warning("No CSV rows match the selected group.")
            return
        n_all = len(csv_rows_all)
        n_f = len(csv_rows)
        count_msg = f"{n_f} row(s)" if n_f == n_all else f"{n_f} of {n_all} row(s) (group filter)"
        st.sidebar.success(f"CSV: {count_msg} — {csv_p.name}")
        st.caption(f"Browsing tomograms from CSV ({csv_p}); group filter: **{csv_group_pick}**.")
    groups = discover_tomograms(data_path)
    if not csv_mode and not groups:
        st.warning("No tomogram groups found. Ensure the data root has subdirs like 15F1/TOP_TOMOS/, 15F1-H4K2Cys/TOP_TOMOS/, etc.")
        return

    selected_group = ""
    tomo_set_pairs: list[tuple[str, str]] = []
    csv_skipped: list[str] = []

    if csv_mode:
        assert csv_rows is not None
        flat_sequence: list[tuple[int, int, int]] = []
        for ri, (tomo_name, set_name, alignment_dir) in enumerate(csv_rows):
            tomo_path = get_tomogram_path(data_path, set_name, tomo_name)
            if not tomo_path:
                csv_skipped.append(f"{tomo_name} ({set_name}): tomogram path missing under data root")
                continue
            sub = tomo_path / alignment_dir
            if not sub.is_dir():
                csv_skipped.append(f"{tomo_name} / {alignment_dir}: alignment directory not found")
                continue
            alignments = get_alignment_dirs_named(tomo_path, alignment_dir)
            if not alignments:
                csv_skipped.append(f"{tomo_name} / {alignment_dir}: no active_zonograms images in expected locations")
                continue
            n_pairs = 0
            for ai, (_, align_az_dir) in enumerate(alignments):
                pairs = get_az_pairs(align_az_dir)
                for pi in range(len(pairs)):
                    flat_sequence.append((ri, ai, pi))
                    n_pairs += 1
            if n_pairs == 0:
                csv_skipped.append(f"{tomo_name} / {alignment_dir}: no active zone PNG pairs")
        if csv_skipped:
            with st.sidebar.expander(f"CSV rows skipped ({len(csv_skipped)})", expanded=False):
                for line in csv_skipped:
                    st.text(line)
    else:
        group_names = sorted(groups.keys())
        selected_group = st.sidebar.selectbox("Group", group_names, index=0)
        tomo_set_pairs = groups[selected_group]

        flat_sequence = []
        for ti, (tomo_name, set_name) in enumerate(tomo_set_pairs):
            tomo_path = get_tomogram_path(data_path, set_name, tomo_name)
            if not tomo_path:
                continue
            alignments = get_alignment_dirs(tomo_path)
            for ai, (_, align_az_dir) in enumerate(alignments):
                pairs = get_az_pairs(align_az_dir)
                for pi in range(len(pairs)):
                    flat_sequence.append((ti, ai, pi))

    n_flat = len(flat_sequence)

    if "flat_pos" not in st.session_state:
        st.session_state.flat_pos = 0
    if csv_mode:
        assert csv_p is not None
        nav_sig = f"csv|{data_path.resolve()}|{csv_p}|{csv_group_pick}"
    else:
        nav_sig = f"dir|{data_path.resolve()}|{selected_group}"
    if st.session_state.get("zv_nav_sig") != nav_sig:
        st.session_state.zv_nav_sig = nav_sig
        st.session_state.flat_pos = 0
    flat_pos = max(0, min(st.session_state.flat_pos, n_flat - 1)) if n_flat else 0

    c1, c2, _ = st.sidebar.columns([1, 1, 2])
    with c1:
        if st.button("←", use_container_width=True, key="prev_nav") and n_flat > 0:
            st.session_state.flat_pos = (flat_pos - 1) % n_flat
            st.rerun()
    with c2:
        if st.button("→", use_container_width=True, key="next_nav") and n_flat > 0:
            st.session_state.flat_pos = (flat_pos + 1) % n_flat
            st.rerun()

    if n_flat == 0:
        if csv_mode:
            st.warning("No active zonograms found for any CSV row (see sidebar for skipped rows).")
        else:
            st.warning("No active zonograms found in any tomogram in this group.")
        return

    seq_idx, align_idx, az_pair_idx = flat_sequence[flat_pos]
    st.session_state.flat_pos = flat_pos

    if csv_mode:
        assert csv_rows is not None
        selected_tomo, set_name, align_from_csv = csv_rows[seq_idx]
        selected_group = set_name
        tomo_path = get_tomogram_path(data_path, set_name, selected_tomo)
        if not tomo_path:
            st.error(f"Tomogram path not found: {set_name}/TOP_TOMOS/{selected_tomo}")
            return
        alignments = get_alignment_dirs_named(tomo_path, align_from_csv)
    else:
        selected_tomo, set_name = tomo_set_pairs[seq_idx]
        tomo_path = get_tomogram_path(data_path, set_name, selected_tomo)
        if not tomo_path:
            st.error(f"Tomogram path not found: {set_name}/TOP_TOMOS/{selected_tomo}")
            return
        alignments = get_alignment_dirs(tomo_path)

    align_names = [a[0] for a in alignments]
    selected_align_idx = min(align_idx, len(alignments) - 1) if alignments else 0
    selected_align_name, az_dir = alignments[selected_align_idx]

    pairs = get_az_pairs(az_dir)
    az_pair_idx = min(az_pair_idx, len(pairs) - 1) if pairs else 0
    idx, pos_path, main_path = pairs[az_pair_idx]
    selected_aunps_path = az_dir / f"active_zonogram_{idx}_selected_aunps.png"
    selected_aunps_manual_path = az_dir / f"active_zonogram_{idx}_selected_aunps_manual.png"

    def find_flat_pos(target_seq: int, target_ai: int, target_pi: int) -> int:
        for i, (si, ai, pi) in enumerate(flat_sequence):
            if si == target_seq and ai == target_ai and pi == target_pi:
                return i
        return flat_pos

    if csv_mode:
        assert csv_rows is not None

        def _csv_row_label(i: int) -> str:
            t, s, a = csv_rows[i]
            return f"{t} / {a} ({s})"

        tomo_select = st.sidebar.selectbox(
            "Tomogram (CSV order)",
            range(len(csv_rows)),
            format_func=_csv_row_label,
            index=seq_idx,
        )
    else:
        tomo_select = st.sidebar.selectbox(
            "Tomogram",
            range(len(tomo_set_pairs)),
            format_func=lambda i: tomo_set_pairs[i][0],
            index=seq_idx,
        )

    align_select = st.sidebar.selectbox(
        "Alignment directory",
        range(len(align_names)),
        format_func=lambda i: align_names[i],
        index=selected_align_idx,
    )
    az_options = [p[0] for p in pairs]
    az_select = st.sidebar.selectbox(
        "Active zone",
        range(len(az_options)),
        format_func=lambda i: str(az_options[i]),
        index=az_pair_idx,
    )

    if tomo_select != seq_idx:
        st.session_state.flat_pos = find_flat_pos(tomo_select, 0, 0)
        st.rerun()
    elif (align_select, az_select) != (selected_align_idx, az_pair_idx):
        new_flat = find_flat_pos(tomo_select, align_select, az_select)
        if new_flat != flat_pos:
            st.session_state.flat_pos = new_flat
            st.rerun()

    # Property assignment (include / improve / exclude) and sample type
    assignments_path = data_path / ASSIGNMENTS_CSV
    assignments = load_assignments(assignments_path)
    az_key = (selected_group, selected_tomo, selected_align_name, str(idx))
    current_row = assignments.get(
        az_key,
        {
            "property": "",
            "sample_type": "",
            "aunp_pick_quality": "",
            "membrane_segmentation_quality": "",
            "redeposition_issues": "",
        },
    )
    current_prop = current_row.get("property", "")
    current_sample_type = current_row.get("sample_type", "")
    current_aunp_pick_quality = current_row.get("aunp_pick_quality", "")
    current_membrane_seg_quality = current_row.get("membrane_segmentation_quality", "")
    current_redeposition_issues = current_row.get("redeposition_issues", "")

    prop_labels = ["— (none)", "include", "improve", "exclude"]
    prop_values = ["", "include", "improve", "exclude"]
    sample_labels = ["— (none)", "tissue", "semi-tissue", "synaptosome"]
    sample_values = ["", "tissue", "semi-tissue", "synaptosome"]
    aunp_pick_quality_labels = ["none", "good", "a few missing", "many missing/wrong"]
    aunp_pick_quality_values = ["", "good", "a few missing", "many missing/wrong"]
    membrane_seg_quality_labels = ["none", "complete", "small issues", "significant issues"]
    membrane_seg_quality_values = ["", "complete", "small issues", "significant issues"]
    redeposition_issues_labels = ["none", "yes", "no"]
    redeposition_issues_values = ["", "yes", "no"]

    prop_key = f"prop_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"
    sample_key = f"sample_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"
    aunp_pick_quality_key = f"aunp_pick_quality_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"
    membrane_seg_quality_key = f"membrane_seg_quality_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"
    redeposition_issues_key = f"redeposition_issues_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"

    def on_property_change():
        radio_idx = st.session_state.get(prop_key, 0)
        new_val = prop_values[radio_idx] if 0 <= radio_idx < len(prop_values) else ""
        save_assignment(
            assignments_path, selected_group, selected_tomo, selected_align_name, str(idx),
            property_value=new_val,
            sample_type_value=current_sample_type,
            aunp_pick_quality_value=current_aunp_pick_quality,
            membrane_segmentation_quality_value=current_membrane_seg_quality,
            redeposition_issues_value=current_redeposition_issues,
        )

    def on_sample_type_change():
        radio_idx = st.session_state.get(sample_key, 0)
        new_val = sample_values[radio_idx] if 0 <= radio_idx < len(sample_values) else ""
        save_assignment(
            assignments_path, selected_group, selected_tomo, selected_align_name, str(idx),
            property_value=current_prop,
            sample_type_value=new_val,
            aunp_pick_quality_value=current_aunp_pick_quality,
            membrane_segmentation_quality_value=current_membrane_seg_quality,
            redeposition_issues_value=current_redeposition_issues,
        )

    def on_aunp_pick_quality_change():
        radio_idx = st.session_state.get(aunp_pick_quality_key, 0)
        new_val = aunp_pick_quality_values[radio_idx] if 0 <= radio_idx < len(aunp_pick_quality_values) else ""
        save_assignment(
            assignments_path,
            selected_group,
            selected_tomo,
            selected_align_name,
            str(idx),
            property_value=current_prop,
            sample_type_value=current_sample_type,
            aunp_pick_quality_value=new_val,
            membrane_segmentation_quality_value=current_membrane_seg_quality,
            redeposition_issues_value=current_redeposition_issues,
        )

    def on_membrane_seg_quality_change():
        radio_idx = st.session_state.get(membrane_seg_quality_key, 0)
        new_val = membrane_seg_quality_values[radio_idx] if 0 <= radio_idx < len(membrane_seg_quality_values) else ""
        save_assignment(
            assignments_path,
            selected_group,
            selected_tomo,
            selected_align_name,
            str(idx),
            property_value=current_prop,
            sample_type_value=current_sample_type,
            aunp_pick_quality_value=current_aunp_pick_quality,
            membrane_segmentation_quality_value=new_val,
            redeposition_issues_value=current_redeposition_issues,
        )

    def on_redeposition_issues_change():
        radio_idx = st.session_state.get(redeposition_issues_key, 0)
        new_val = redeposition_issues_values[radio_idx] if 0 <= radio_idx < len(redeposition_issues_values) else ""
        save_assignment(
            assignments_path,
            selected_group,
            selected_tomo,
            selected_align_name,
            str(idx),
            property_value=current_prop,
            sample_type_value=current_sample_type,
            aunp_pick_quality_value=current_aunp_pick_quality,
            membrane_segmentation_quality_value=current_membrane_seg_quality,
            redeposition_issues_value=new_val,
        )

    prop_idx = prop_values.index(current_prop) if current_prop in prop_values else 0
    sample_idx = sample_values.index(current_sample_type) if current_sample_type in sample_values else 0
    aunp_pick_quality_idx = (
        aunp_pick_quality_values.index(current_aunp_pick_quality)
        if current_aunp_pick_quality in aunp_pick_quality_values
        else 0
    )
    membrane_seg_quality_idx = (
        membrane_seg_quality_values.index(current_membrane_seg_quality)
        if current_membrane_seg_quality in membrane_seg_quality_values
        else 0
    )
    redeposition_issues_idx = (
        redeposition_issues_values.index(current_redeposition_issues)
        if current_redeposition_issues in redeposition_issues_values
        else 0
    )

    st.header("Current view")
    st.write(f"**Tomogram:** {selected_tomo}")
    st.write(f"**Alignment directory:** {selected_align_name}")
    st.write(f"**Active Zone:** {idx}")

    st.subheader("Data curation")
    st.radio(
        "Include / Improve / Exclude",
        range(len(prop_labels)),
        format_func=lambda i: prop_labels[i],
        index=prop_idx,
        horizontal=True,
        key=prop_key,
        on_change=on_property_change,
    )

    st.radio(
        "Morphology",
        range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
        index=sample_idx,
        horizontal=True,
        key=sample_key,
        on_change=on_sample_type_change,
    )
    st.radio(
        "AuNP pick quality",
        range(len(aunp_pick_quality_labels)),
        format_func=lambda i: aunp_pick_quality_labels[i],
        index=aunp_pick_quality_idx,
        horizontal=True,
        key=aunp_pick_quality_key,
        on_change=on_aunp_pick_quality_change,
    )
    st.radio(
        "Membrane segmentation quality",
        range(len(membrane_seg_quality_labels)),
        format_func=lambda i: membrane_seg_quality_labels[i],
        index=membrane_seg_quality_idx,
        horizontal=True,
        key=membrane_seg_quality_key,
        on_change=on_membrane_seg_quality_change,
    )
    st.radio(
        "Redeposition issues",
        range(len(redeposition_issues_labels)),
        format_func=lambda i: redeposition_issues_labels[i],
        index=redeposition_issues_idx,
        horizontal=True,
        key=redeposition_issues_key,
        on_change=on_redeposition_issues_change,
    )
    if (
        current_prop
        or current_sample_type
        or current_aunp_pick_quality
        or current_membrane_seg_quality
        or current_redeposition_issues
    ):
        parts = [
            p
            for p in [
                current_prop,
                current_sample_type,
                current_aunp_pick_quality,
                current_membrane_seg_quality,
                current_redeposition_issues,
            ]
            if p
        ]
        st.caption(f"Current: {' | '.join(parts)}  (output: {assignments_path})")
    st.caption(str(az_dir))

    col1, col2 = st.columns(2)
    with col1:
        if pos_path and pos_path.exists():
            st.subheader("Position")
            st.image(str(pos_path), use_container_width=True)
        else:
            st.subheader("Position")
            st.info("No position image found")

        st.subheader("Tomogram slice")
        slice_options = ["-2", "-1", "central", "+1", "+2"]
        slice_mode = st.select_slider(
            "Slice position",
            options=slice_options,
            value="central",
            key=f"slice_mode_{selected_group}_{selected_tomo}_{selected_align_name}",
        )
        alignment_dir = infer_alignment_dir_from_az_dir(az_dir, tomo_path)
        if alignment_dir is None:
            st.info("Could not infer alignment directory for tomogram slice.")
        else:
            ddw_mrc_path = get_ddw_mrc_path(alignment_dir)
            if ddw_mrc_path is None:
                st.info("No *ddw.mrc found for this alignment.")
            else:
                try:
                    slice_png = ensure_tomogram_slice_png(ddw_mrc_path, slice_mode)
                    st.image(str(slice_png), use_container_width=True)
                except Exception as exc:
                    st.info(f"Could not load tomogram slice: {exc}")
    with col2:
        if main_path and main_path.exists():
            st.subheader("Zonogram")
            st.image(str(main_path), use_container_width=True)
        else:
            st.subheader("Zonogram")
            st.info("No main zonogram image found")

        if selected_aunps_path.exists():
            st.subheader("Selected AuNPs")
            st.image(str(selected_aunps_path), use_container_width=True)

        if selected_aunps_manual_path.exists():
            st.subheader("Selected AuNPs (Manual)")
            st.image(str(selected_aunps_manual_path), use_container_width=True)

    # Quick navigation
    st.sidebar.markdown("---")
    st.sidebar.caption("Navigate with the dropdowns above.")


if __name__ == "__main__":
    main()
