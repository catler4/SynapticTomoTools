#!/usr/bin/env python3
"""
Streamlit-based Active Zonogram Viewer

Quickly browse active zonogram images (position + main) across tomogram groups,
tomograms, and alignment directories (best_alignment, liza_az0, liza_az1, etc.).

Usage:
    streamlit run scripts/zonogram_viewer.py
    streamlit run scripts/zonogram_viewer.py --server.address 0.0.0.0 --server.port 8501

Or with custom data path:
    streamlit run scripts/zonogram_viewer.py -- --data-dir /path/to/data
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ASSIGNMENTS_CSV = "zonogram_viewer_assignments.csv"
PROPERTY_OPTIONS = ["", "include", "improve", "exclude"]
# Sample/morphology options (alternatives: strongly/moderately tissue-like, tissue-like/partially tissue-like)
SAMPLE_TYPE_OPTIONS = ["", "tissue", "semi-tissue", "synaptosome"]

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    """Load assignments from CSV. Key = (group, tomoname, alignment_dir, active_zone), value = {property, sample_type}."""
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
        "updated_at": now,
    }
    rows = sorted(rows_dict.values(), key=lambda r: (r["group"], r["tomoname"], r["alignment_directory"], r["active_zone"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "tomoname", "alignment_directory", "active_zone", "property", "sample_type", "updated_at"])
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

    # Discover groups
    groups = discover_tomograms(data_path)
    if not groups:
        st.warning("No tomogram groups found. Ensure the data root has subdirs like 15F1/TOP_TOMOS/, 15F1-H4K2Cys/TOP_TOMOS/, etc.")
        return

    group_names = sorted(groups.keys())
    selected_group = st.sidebar.selectbox("Group", group_names, index=0)
    tomo_set_pairs = groups[selected_group]

    # Build flat sequence: for each tomogram, each alignment, each active zone
    # Order: tomo0 align0 az0, tomo0 align0 az1, ..., tomo0 align1 az0, ..., tomo1 align0 az0, ...
    flat_sequence: list[tuple[int, int, int]] = []
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

    # Session state for flat position (Prev/Next cycles through alignment→AZ within tomogram, then next tomogram)
    if "flat_pos" not in st.session_state:
        st.session_state.flat_pos = 0
    if st.session_state.get("last_group") != selected_group:
        st.session_state.last_group = selected_group
        st.session_state.flat_pos = 0
    flat_pos = max(0, min(st.session_state.flat_pos, n_flat - 1)) if n_flat else 0

    # Prev / Next buttons
    c1, c2, _ = st.sidebar.columns([1, 1, 2])
    with c1:
        if st.button("←", use_container_width=True, key="prev_nav") and n_flat > 0:
            st.session_state.flat_pos = (flat_pos - 1) % n_flat
            st.rerun()
    with c2:
        if st.button("→", use_container_width=True, key="next_nav") and n_flat > 0:
            st.session_state.flat_pos = (flat_pos + 1) % n_flat
            st.rerun()

    # Resolve current position from flat sequence
    if n_flat == 0:
        st.warning("No active zonograms found in any tomogram in this group.")
        return
    tomo_idx, align_idx, az_pair_idx = flat_sequence[flat_pos]
    st.session_state.flat_pos = flat_pos

    selected_tomo, set_name = tomo_set_pairs[tomo_idx]
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

    # Dropdowns for manual navigation; changing them updates flat_pos
    def find_flat_pos(target_ti: int, target_ai: int, target_pi: int) -> int:
        for i, (ti, ai, pi) in enumerate(flat_sequence):
            if ti == target_ti and ai == target_ai and pi == target_pi:
                return i
        return flat_pos  # keep current if no match (e.g. different tomogram structure)

    tomo_select = st.sidebar.selectbox(
        "Tomogram",
        range(len(tomo_set_pairs)),
        format_func=lambda i: tomo_set_pairs[i][0],
        index=tomo_idx,
    )
    # Alignments and pairs are for current tomogram
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
    # Sync flat_pos if user changed dropdowns
    if tomo_select != tomo_idx:
        # Different tomogram: jump to first (align 0, az 0) of that tomogram
        new_flat = find_flat_pos(tomo_select, 0, 0)
        st.session_state.flat_pos = new_flat
        st.rerun()
    elif (align_select, az_select) != (selected_align_idx, az_pair_idx):
        # Same tomogram, different align/az
        new_flat = find_flat_pos(tomo_select, align_select, az_select)
        if new_flat != flat_pos:
            st.session_state.flat_pos = new_flat
            st.rerun()

    # Property assignment (include / improve / exclude) and sample type
    assignments_path = data_path / ASSIGNMENTS_CSV
    assignments = load_assignments(assignments_path)
    az_key = (selected_group, selected_tomo, selected_align_name, str(idx))
    current_row = assignments.get(az_key, {"property": "", "sample_type": ""})
    current_prop = current_row.get("property", "")
    current_sample_type = current_row.get("sample_type", "")

    prop_labels = ["— (none)", "include", "improve", "exclude"]
    prop_values = ["", "include", "improve", "exclude"]
    sample_labels = ["— (none)", "tissue", "semi-tissue", "synaptosome"]
    sample_values = ["", "tissue", "semi-tissue", "synaptosome"]

    prop_key = f"prop_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"
    sample_key = f"sample_radio_{selected_group}_{selected_tomo}_{selected_align_name}_{idx}"

    def on_property_change():
        radio_idx = st.session_state.get(prop_key, 0)
        new_val = prop_values[radio_idx] if 0 <= radio_idx < len(prop_values) else ""
        save_assignment(
            assignments_path, selected_group, selected_tomo, selected_align_name, str(idx),
            property_value=new_val, sample_type_value=current_sample_type,
        )

    def on_sample_type_change():
        radio_idx = st.session_state.get(sample_key, 0)
        new_val = sample_values[radio_idx] if 0 <= radio_idx < len(sample_values) else ""
        save_assignment(
            assignments_path, selected_group, selected_tomo, selected_align_name, str(idx),
            property_value=current_prop, sample_type_value=new_val,
        )

    prop_idx = prop_values.index(current_prop) if current_prop in prop_values else 0
    sample_idx = sample_values.index(current_sample_type) if current_sample_type in sample_values else 0

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

    st.subheader("Sample properties")
    st.radio(
        "Morphology",
        range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
        index=sample_idx,
        horizontal=True,
        key=sample_key,
        on_change=on_sample_type_change,
    )
    if current_prop or current_sample_type:
        parts = [p for p in [current_prop, current_sample_type] if p]
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

    # Quick navigation
    st.sidebar.markdown("---")
    st.sidebar.caption("Navigate with the dropdowns above.")


if __name__ == "__main__":
    main()
