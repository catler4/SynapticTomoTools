#!/usr/bin/env python3
"""
Assemble a CZII public-upload tree from processed tomograms + raw frames.

Reads a tomogram CSV (``tomoname``, ``set``, ``alignment_dir``) and copies:

  {dest}/{set}/{tomoname}/
    frames/                         # raw movies + mdocs + gain (once per tomo)
    {renamed_alignment}/
      {tomoname}.defocus
      {tomoname}.xf                 # alignment transforms (most sets)
      {tomoname}.mrc                # motion-corrected tilt stack (most sets)
      {tomoname}.xml / {tomoname}.st  # H4K2Cys only (Warp etomo + tiltstack)
      {tomoname}_full_rec_BP_3DCTF_BIN4.mrc
      {tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc
      annotations/
        AuNP pick STARs, MemBrain smooth seg, pre/post .glb

Alignment rename:
  best_alignment      -> alignment_IMOD
  liza_azN[_rerun]    -> alignment_cleftN

For set ``15F1-H4K2Cys`` only: skip ``.xf`` / ``.mrc``; copy Warp
``{tomoname}.xml`` (etomo) and ``{tomoname}.st`` (tiltstack/{tomoname}/)
instead.

Example
-------
python scripts/utilities/prepare_czii_data_upload.py \\
    --tomocsv tomogram_csv_files/tomograms_full_set_FINAL.csv \\
    --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROCESSED_ROOT = Path(
    "/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms"
)
DEFAULT_RAW_ROOT = Path("/home/groups/mammoth/rawdata/CJS/BestTomo_rawdata")
DEFAULT_DEST_ROOT = Path("/goliath/rawdata/Gouaux/CJS/TOMO/CZII_data_upload")
DEFAULT_H4K2CYS_ETOMO_DIR = Path(
    "/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/analyses/"
    "20260324_H4K2Cys_TOMO_Warp_CJS/warp_tiltseries/etomo"
)
DEFAULT_H4K2CYS_TILTSTACK_ROOT = Path(
    "/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/analyses/"
    "20260324_H4K2Cys_TOMO_Warp_CJS/warp_tiltseries_2.5/tiltstack"
)

FRAME_SUFFIXES = (".tif", ".eer", ".mrc")
GAIN_NAME_RE = re.compile(r"^gain_reference\.", re.IGNORECASE)
LIZA_AZ_RE = re.compile(r"^liza_az(\d+)(?:_rerun)?$", re.IGNORECASE)
POSITION_RE = re.compile(r"(Position[_\-][\w\-]+)", re.IGNORECASE)
DIMER_SET = "15F1and5F11dimer"
UNLABELED_SET = "unlabeled"
H4K2CYS_SET = "15F1-H4K2Cys"

# AuNP pick patterns (basename globs under aunps/)
AUNP_STAR_DEFAULT = "aunp_tm_BP_active_zone_*_manual_refined.star"
AUNP_STAR_MONOMER = "aunp_tm_BP_active_zone_*_manual_refined_monomer.star"
AUNP_STAR_DIMER = "aunp_tm_BP_active_zone_*_manual_refined_dimer.star"
# Also accept hyphen variant if present on disk
AUNP_STAR_MONOMER_ALT = "aunp_tm_BP_active_zone_*_manual_refined-monomer.star"
AUNP_STAR_DIMER_ALT = "aunp_tm_BP_active_zone_*_manual_refined-dimer.star"

XYZ_COLS = ("faCoordinateX", "faCoordinateY", "faCoordinateZ")
XYZ_STAR_SUFFIX = "_xyz_only"  # inserted before .star; originals are kept


@dataclass
class CopyPlan:
    src: Path
    dest: Path
    kind: str


@dataclass
class TomogramJob:
    set_name: str
    tomoname: str
    alignment_dirs: list[str] = field(default_factory=list)


def map_alignment_dir(alignment_dir: str) -> str:
    name = alignment_dir.strip()
    if name == "best_alignment":
        return "alignment_IMOD"
    m = LIZA_AZ_RE.match(name)
    if m:
        return f"alignment_cleft{int(m.group(1))}"
    raise ValueError(
        f"Unrecognized alignment_dir {alignment_dir!r}; "
        "expected best_alignment or liza_azN[_rerun]"
    )


def load_jobs(csv_path: Path, sets_filter: set[str] | None) -> list[TomogramJob]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"tomoname", "set", "alignment_dir"}
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError(f"{csv_path} must have columns {sorted(required)}")
        for row in reader:
            tomoname = (row.get("tomoname") or "").strip()
            set_name = (row.get("set") or "").strip()
            alignment = (row.get("alignment_dir") or "").strip()
            if not tomoname or not set_name or tomoname.startswith("---"):
                continue
            if sets_filter and set_name not in sets_filter:
                continue
            if not alignment or alignment.lower() in ("nan", "none"):
                continue
            key = (set_name, tomoname)
            if alignment not in grouped[key]:
                grouped[key].append(alignment)
    jobs = [
        TomogramJob(set_name=s, tomoname=t, alignment_dirs=aligns)
        for (s, t), aligns in grouped.items()
    ]
    jobs.sort(key=lambda j: (j.set_name, j.tomoname))
    return jobs


def position_token(tomoname: str) -> str | None:
    """Return Position_… token used to match EER/mdoc basenames, if any."""
    m = POSITION_RE.search(tomoname)
    return m.group(1) if m else None


def position_basename(tomoname: str) -> str | None:
    """
    Basename starting at ``Position_…`` through the end of ``tomoname``.

    Example: ``AMmilled49-1_Position_10_2`` → ``Position_10_2``.
    Used when on-disk files still use the shorter original name.
    """
    m = POSITION_RE.search(tomoname)
    if not m:
        return None
    short = tomoname[m.start() :]
    return short if short and short != tomoname else None


def session_tokens(tomoname: str) -> list[str]:
    """
    Session identifiers that must contain a Position_…-only raw match.

    Example: ``20250418_AMmilled29-2_Position_47`` →
    ``['20250418_AMmilled29-2', 'AMmilled29-2']``.
    Multiple sessions can reuse the same Position_ labels, so short-name
    frame matches are only accepted under a matching session directory.
    """
    m = POSITION_RE.search(tomoname)
    if not m:
        return []
    prefix = tomoname[: m.start()].rstrip("_-")
    if not prefix:
        return []
    tokens = [prefix]
    date_stripped = re.match(r"^\d{8}_(.+)$", prefix)
    if date_stripped and date_stripped.group(1) not in tokens:
        tokens.append(date_stripped.group(1))
    return tokens


def path_matches_session(path: Path, sessions: list[str]) -> bool:
    """True if any path component equals or is prefixed by a session token."""
    if not sessions:
        return False
    for part in path.parts:
        for session in sessions:
            if part == session or part.startswith(session + "_") or part.startswith(
                session + "-"
            ):
                return True
    return False


def short_position_filename(dest_name: str, tomoname: str) -> str | None:
    """Map a full-tomoname filename to the Position_… basename variant."""
    short = position_basename(tomoname)
    if not short or tomoname not in dest_name:
        return None
    return dest_name.replace(tomoname, short, 1)


def resolve_alignment_file(
    align_path: Path, tomoname: str, dest_name: str
) -> Path | None:
    """
    Find ``dest_name`` under the alignment dir (or parent for tilt stack).

    If the full tomoname file is missing and ``tomoname`` contains ``Position_…``,
    also try the shorter basename that starts at ``Position_``. Destination naming
    still uses ``dest_name`` (full tomoname).
    """
    candidates = [align_path / dest_name]
    # Motion-corrected tilt stack sometimes lives one level up (tomo root)
    if dest_name == f"{tomoname}.mrc":
        candidates.append(align_path.parent / dest_name)

    short_name = short_position_filename(dest_name, tomoname)
    if short_name and short_name != dest_name:
        candidates.append(align_path / short_name)
        if dest_name == f"{tomoname}.mrc":
            candidates.append(align_path.parent / short_name)

    for path in candidates:
        if path.exists():
            return path
    return None


def is_frame_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if GAIN_NAME_RE.match(name):
        return False
    if name.lower().endswith(".mdoc"):
        return False
    lower = name.lower()
    return any(lower.endswith(suf) for suf in FRAME_SUFFIXES)


def is_mdoc_file(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(".mdoc")


def is_gain_file(path: Path) -> bool:
    return path.is_file() and GAIN_NAME_RE.match(path.name) is not None


def frame_matches_tomo(
    path: Path,
    tomoname: str,
    pos: str | None,
    sessions: list[str] | None = None,
) -> bool:
    """
    Match a raw frame/mdoc to ``tomoname``.

    Full-name / date-stripped matches are accepted anywhere. A match that only
    uses the ``Position_…`` token also requires the file to sit under a session
    directory matching ``sessions`` (so ``Position_20`` in session A is not
    confused with ``Position_20`` in session B).
    """
    stem = path.name
    if tomoname in stem:
        return True
    # Also match without date prefix: strip leading YYYYMMDD_ if present
    m = re.match(r"^\d{8}_(.+)$", tomoname)
    if m and m.group(1) in stem:
        return True
    if pos and pos in stem:
        if not sessions:
            # No session context available → refuse Position-only match
            return False
        return path_matches_session(path, sessions)
    return False


def find_named_dirs(root: Path, name: str) -> list[Path]:
    """Find directories named ``name`` under root (bounded search)."""
    hits: list[Path] = []
    if not root.is_dir():
        return hits
    # Direct / one-level-nested patterns first
    candidates = [root / name]
    for child in root.iterdir():
        if child.is_dir():
            candidates.append(child / name)
    for candidate in candidates:
        if candidate.is_dir() and candidate.name == name:
            hits.append(candidate)
    if hits:
        return sorted(set(hits))
    # Broader walk for deeper nesting
    for dirpath, dirnames, _filenames in os_walk_limited(root, max_depth=4):
        if Path(dirpath).name == name:
            hits.append(Path(dirpath))
            dirnames.clear()
    return sorted(set(hits))


def os_walk_limited(root: Path, max_depth: int):
    """os.walk-like generator with a max depth from root."""
    root = root.resolve()
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in __import__("os").walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= max_depth:
            dirnames.clear()
        yield dirpath, dirnames, filenames


def collect_from_directory(
    directory: Path, tomoname: str, sessions: list[str] | None = None
) -> tuple[list[Path], list[Path], list[Path]]:
    """Collect frames/mdocs/gain from a tilt-series directory (layout A or mixed)."""
    pos = position_token(tomoname)
    if sessions is None:
        sessions = session_tokens(tomoname)
    frames: list[Path] = []
    mdocs: list[Path] = []
    gains: list[Path] = []

    search_roots = [directory]
    frames_subdir = directory / "frames"
    if frames_subdir.is_dir():
        search_roots.append(frames_subdir)
    mdoc_subdir = directory / "mdoc"
    if mdoc_subdir.is_dir():
        search_roots.append(mdoc_subdir)
    # Parent session may hold shared gain / mdoc
    parent = directory.parent
    if parent.is_dir():
        if (parent / "mdoc").is_dir():
            search_roots.append(parent / "mdoc")
        if (parent / "frames").is_dir() and parent / "frames" not in search_roots:
            search_roots.append(parent / "frames")
        search_roots.append(parent)

    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            if is_gain_file(path):
                # Prefer gain next to the series; still accept session-level
                seen.add(resolved)
                gains.append(path)
                continue
            if is_mdoc_file(path):
                if (
                    frame_matches_tomo(path, tomoname, pos, sessions)
                    or path.parent == directory
                ):
                    seen.add(resolved)
                    mdocs.append(path)
                continue
            if is_frame_file(path) and frame_matches_tomo(
                path, tomoname, pos, sessions
            ):
                seen.add(resolved)
                frames.append(path)

    # If directory itself is the series folder and contains unmatched frames that
    # all share the tomoname / Position_ prefix, take all non-gain frames/mdocs
    # in that folder. Directory was already session-scoped by the caller.
    if not frames and directory.is_dir():
        local_frames = [
            p
            for p in directory.iterdir()
            if p.is_file() and is_frame_file(p)
        ]
        if local_frames and all(
            tomoname in p.name or (pos and pos in p.name) for p in local_frames
        ):
            frames.extend(local_frames)
            mdocs.extend(
                p
                for p in directory.iterdir()
                if p.is_file() and is_mdoc_file(p)
            )
            gains.extend(
                p for p in directory.iterdir() if p.is_file() and is_gain_file(p)
            )

    return sorted(set(frames)), sorted(set(mdocs)), sorted(set(gains))


def find_session_dirs(set_root: Path, sessions: list[str]) -> list[Path]:
    """Find directories under set_root whose name matches a session token."""
    if not sessions or not set_root.is_dir():
        return []
    hits: list[Path] = []
    for dirpath, dirnames, _filenames in os_walk_limited(set_root, max_depth=3):
        p = Path(dirpath)
        if p == set_root:
            continue
        for session in sessions:
            if p.name == session or p.name.startswith(session + "_") or p.name.startswith(
                session + "-"
            ):
                hits.append(p)
                break
    return sorted(set(hits))


def find_raw_assets(
    raw_root: Path, set_name: str, tomoname: str
) -> tuple[list[Path], list[Path], list[Path], list[str]]:
    """
    Locate raw frames + mdocs + gain for one tomogram.

    Supports:
    - Series folder with per-tilt .tif/.eer/.mrc (+ .mdoc)
    - Session folder with frames/ + mdoc/ (EER style)

    ``Position_…``-only names are only accepted inside a matching session
    directory (prefix of ``tomoname`` before ``Position_``).
    """
    warnings: list[str] = []
    set_root = raw_root / set_name
    if not set_root.is_dir():
        return [], [], [], [f"raw set root missing: {set_root}"]

    pos = position_token(tomoname)
    short = position_basename(tomoname)
    sessions = session_tokens(tomoname)

    # 1) Directory named as full tomoname (unambiguous)
    named = find_named_dirs(set_root, tomoname)

    # 2) Position_…-only series dirs — only under a matching session
    if short and short != tomoname and sessions:
        for session_dir in find_session_dirs(set_root, sessions):
            for d in find_named_dirs(session_dir, short):
                if d not in named:
                    named.append(d)
            # Session dir itself may be the series folder named Position_…
            if session_dir.name == short and session_dir not in named:
                named.append(session_dir)

    frames: list[Path] = []
    mdocs: list[Path] = []
    gains: list[Path] = []

    for d in named:
        f, m, g = collect_from_directory(d, tomoname, sessions)
        frames.extend(f)
        mdocs.extend(m)
        gains.extend(g)

    # 3) If nothing found, search frames/ folders (Position-only matches
    #    still require a matching session path component)
    if not frames:
        for dirpath, dirnames, _filenames in os_walk_limited(set_root, max_depth=3):
            p = Path(dirpath)
            if p.name != "frames":
                continue
            local_frames = [
                path
                for path in p.iterdir()
                if path.is_file()
                and is_frame_file(path)
                and frame_matches_tomo(path, tomoname, pos, sessions)
            ]
            if not local_frames:
                continue
            frames.extend(local_frames)
            # sibling mdoc/ + per-frame mdocs next to movies
            mdoc_dir = p.parent / "mdoc"
            mdoc_roots = [p]
            if mdoc_dir.is_dir():
                mdoc_roots.append(mdoc_dir)
            for root in mdoc_roots:
                for path in root.iterdir():
                    if path.is_file() and is_mdoc_file(path) and frame_matches_tomo(
                        path, tomoname, pos, sessions
                    ):
                        mdocs.append(path)
            for gpath in p.parent.iterdir():
                if gpath.is_file() and is_gain_file(gpath):
                    gains.append(gpath)

    frames = sorted(set(frames))
    mdocs = sorted(set(mdocs))
    gains = sorted(set(gains))

    if not frames:
        warnings.append(f"no raw frames found for {set_name}/{tomoname} under {set_root}")
    if not mdocs:
        warnings.append(f"no mdoc files found for {set_name}/{tomoname}")
    if not gains:
        warnings.append(f"no gain_reference found for {set_name}/{tomoname}")
    return frames, mdocs, gains, warnings


def resolve_processed_alignment(
    processed_root: Path, set_name: str, tomoname: str, alignment_dir: str
) -> Path | None:
    path = processed_root / set_name / "TOP_TOMOS" / tomoname / alignment_dir
    if path.is_dir():
        return path
    return None


def resolve_file_with_position_fallback(
    directory: Path, tomoname: str, dest_name: str
) -> Path | None:
    """Look for ``dest_name`` in ``directory``, then the Position_… short name."""
    candidates = [directory / dest_name]
    short_name = short_position_filename(dest_name, tomoname)
    if short_name and short_name != dest_name:
        candidates.append(directory / short_name)
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_h4k2cys_warp_assets(
    tomoname: str,
    etomo_dir: Path,
    tiltstack_root: Path,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """
    For H4K2Cys: locate Warp etomo XML + tiltstack .st (instead of .xf / .mrc).

    Destination names always use the full ``tomoname``.
    """
    warnings: list[str] = []
    items: list[tuple[Path, str]] = []

    xml_dest = f"{tomoname}.xml"
    xml_src = resolve_file_with_position_fallback(etomo_dir, tomoname, xml_dest)
    if xml_src is not None:
        items.append((xml_src, xml_dest))
    else:
        warnings.append(f"missing H4K2Cys etomo XML: {etomo_dir / xml_dest}")

    st_dest = f"{tomoname}.st"
    short = position_basename(tomoname)
    st_dir_candidates = [tiltstack_root / tomoname]
    if short and short != tomoname:
        st_dir_candidates.append(tiltstack_root / short)

    st_src: Path | None = None
    for st_dir in st_dir_candidates:
        st_src = resolve_file_with_position_fallback(st_dir, tomoname, st_dest)
        if st_src is not None:
            break
    if st_src is not None:
        items.append((st_src, st_dest))
    else:
        warnings.append(
            f"missing H4K2Cys tiltstack .st under {tiltstack_root}/{{{tomoname}"
            + (f"|{short}" if short and short != tomoname else "")
            + f"}}/{st_dest}"
        )

    return items, warnings


def collect_alignment_assets(
    align_path: Path,
    tomoname: str,
    set_name: str,
    *,
    h4k2cys_etomo_dir: Path | None = None,
    h4k2cys_tiltstack_root: Path | None = None,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """
    Return (src, relative_dest_under_alignment_dir) pairs and warnings.

    relative_dest is e.g. ``tomoname.defocus`` or ``annotations/foo.star``.
    For ``15F1-H4K2Cys``, ``.xf`` / ``.mrc`` are omitted and Warp ``.xml`` /
    ``.st`` are added instead.
    """
    warnings: list[str] = []
    items: list[tuple[Path, str]] = []
    is_h4k2cys = set_name == H4K2CYS_SET

    required = [
        f"{tomoname}.defocus",
        f"{tomoname}_full_rec_BP_3DCTF_BIN4.mrc",
        f"{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc",
    ]
    if not is_h4k2cys:
        # Insert after defocus to keep a stable order with other sets
        required = [
            f"{tomoname}.defocus",
            f"{tomoname}.xf",
            f"{tomoname}.mrc",
            f"{tomoname}_full_rec_BP_3DCTF_BIN4.mrc",
            f"{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc",
        ]

    for name in required:
        src = resolve_alignment_file(align_path, tomoname, name)
        if src is not None:
            # Always write under the full tomoname in the upload tree
            items.append((src, name))
            continue
        warnings.append(f"missing {align_path / name}")

    if is_h4k2cys:
        etomo = h4k2cys_etomo_dir or DEFAULT_H4K2CYS_ETOMO_DIR
        tiltstack = h4k2cys_tiltstack_root or DEFAULT_H4K2CYS_TILTSTACK_ROOT
        warp_items, warp_warns = resolve_h4k2cys_warp_assets(
            tomoname, etomo, tiltstack
        )
        items.extend(warp_items)
        warnings.extend(warp_warns)

    # Annotations
    ann_sources: list[Path] = []
    aunps_dir = align_path / "aunps"
    membrain_dir = align_path / "membrain"
    expect_aunps = set_name != UNLABELED_SET

    if expect_aunps:
        if set_name == DIMER_SET:
            patterns = [
                AUNP_STAR_MONOMER,
                AUNP_STAR_DIMER,
                AUNP_STAR_MONOMER_ALT,
                AUNP_STAR_DIMER_ALT,
            ]
        else:
            patterns = [AUNP_STAR_DEFAULT]

        if aunps_dir.is_dir():
            for pat in patterns:
                ann_sources.extend(sorted(aunps_dir.glob(pat)))
            for glb in ("presynapticmembranes.glb", "postsynapticmembranes.glb"):
                g = aunps_dir / glb
                if g.is_file():
                    ann_sources.append(g)
                else:
                    warnings.append(f"missing {g}")
        else:
            warnings.append(f"missing aunps/ under {align_path}")

        if set_name == DIMER_SET:
            if not any("monomer" in p.name for p in ann_sources if p.suffix == ".star"):
                warnings.append(f"no monomer STAR found under {aunps_dir}")
            if not any("dimer" in p.name for p in ann_sources if p.suffix == ".star"):
                warnings.append(f"no dimer STAR found under {aunps_dir}")
        else:
            if not any(p.suffix == ".star" for p in ann_sources):
                warnings.append(f"no manual_refined STAR found under {aunps_dir}")
    else:
        # unlabeled: still copy membrane .glb files if present, but do not require AuNP STARs
        if aunps_dir.is_dir():
            for glb in ("presynapticmembranes.glb", "postsynapticmembranes.glb"):
                g = aunps_dir / glb
                if g.is_file():
                    ann_sources.append(g)

    if membrain_dir.is_dir():
        smooth = sorted(membrain_dir.glob("*_smooth.mrc"))
        if smooth:
            ann_sources.extend(smooth)
        else:
            warnings.append(f"no *_smooth.mrc under {membrain_dir}")
    else:
        warnings.append(f"missing membrain/ under {align_path}")

    for src in ann_sources:
        items.append((src, f"annotations/{src.name}"))

    return items, warnings


def plan_copy(src: Path, dest: Path, kind: str, force: bool) -> CopyPlan | None:
    if dest.exists() and not force:
        return None
    return CopyPlan(src=src, dest=dest, kind=kind)


def execute_copy(plan: CopyPlan, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] {plan.kind}: {plan.src} -> {plan.dest}")
        return
    plan.dest.parent.mkdir(parents=True, exist_ok=True)
    # Follow symlinks to real file content for upload
    src = plan.src
    if src.is_symlink():
        src = src.resolve()
    shutil.copy2(src, plan.dest)
    print(f"  copied {plan.kind}: {plan.dest}")


def _star_dataframe(path: Path):
    """Return the first table DataFrame from a Relion/STAR file."""
    import starfile

    star_data = starfile.read(path)
    if isinstance(star_data, dict):
        for value in star_data.values():
            if hasattr(value, "columns"):
                return value
    if hasattr(star_data, "columns"):
        return star_data
    raise ValueError(f"No table found in STAR file: {path}")


def xyz_only_star_path(star_path: Path) -> Path:
    """``foo.star`` -> ``foo_xyz_only.star`` (never overwrites the input name)."""
    if star_path.name.endswith(f"{XYZ_STAR_SUFFIX}.star"):
        return star_path
    return star_path.with_name(f"{star_path.stem}{XYZ_STAR_SUFFIX}.star")


def write_aunp_xyz_only_star(
    src_star: Path,
    *,
    out_path: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Path | None:
    """
    Write a Relion-style STAR containing only faCoordinateX/Y/Z.

    Original ``src_star`` is left unchanged. Returns the output path, or None if
    skipped / failed.
    """
    if src_star.name.endswith(f"{XYZ_STAR_SUFFIX}.star"):
        return None
    if out_path is None:
        out_path = xyz_only_star_path(src_star)
    if out_path.exists() and not force and not dry_run:
        print(f"  skip existing xyz STAR: {out_path}")
        return out_path

    if dry_run:
        print(f"  [dry-run] xyz STAR: {src_star} -> {out_path}")
        return out_path

    if not src_star.is_file():
        print(f"  ERROR: STAR not found for xyz export: {src_star}")
        return None

    try:
        df = _star_dataframe(src_star)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR reading {src_star}: {exc}")
        return None

    # Accept columns with or without a leading underscore
    colmap: dict[str, str] = {}
    for want in XYZ_COLS:
        if want in df.columns:
            colmap[want] = want
        elif f"_{want}" in df.columns:
            colmap[want] = f"_{want}"
        else:
            print(f"  ERROR: {src_star} missing column {want}")
            return None

    slim = df[[colmap[c] for c in XYZ_COLS]].copy()
    slim.columns = list(XYZ_COLS)

    import starfile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(slim, out_path, overwrite=True)
    print(f"  wrote xyz STAR: {out_path}")
    return out_path


def annotation_stars_for_job(dest_tomo: Path) -> list[Path]:
    """Find AuNP annotation STAR files under a destination tomogram tree."""
    stars: list[Path] = []
    if not dest_tomo.is_dir():
        return stars
    for ann_dir in dest_tomo.glob("*/annotations"):
        if not ann_dir.is_dir():
            continue
        for path in sorted(ann_dir.glob("*.star")):
            if path.name.endswith(f"{XYZ_STAR_SUFFIX}.star"):
                continue
            # Only AuNP pick-like STARs
            name = path.name
            if "manual_refined" in name or name.startswith("aunp_tm_BP_active_zone_"):
                stars.append(path)
    return stars


def build_plans_for_job(
    job: TomogramJob,
    *,
    processed_root: Path,
    raw_root: Path,
    dest_root: Path,
    force: bool,
    h4k2cys_etomo_dir: Path | None = None,
    h4k2cys_tiltstack_root: Path | None = None,
) -> tuple[list[CopyPlan], list[str]]:
    plans: list[CopyPlan] = []
    warnings: list[str] = []
    dest_tomo = dest_root / job.set_name / job.tomoname

    # Frames (once)
    frames, mdocs, gains, raw_warns = find_raw_assets(raw_root, job.set_name, job.tomoname)
    warnings.extend(raw_warns)
    frames_dest = dest_tomo / "frames"
    for src in frames + mdocs + gains:
        dest = frames_dest / src.name
        plan = plan_copy(src, dest, "frames", force)
        if plan:
            plans.append(plan)

    # Alignments
    for alignment_dir in job.alignment_dirs:
        try:
            out_name = map_alignment_dir(alignment_dir)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        align_src = resolve_processed_alignment(
            processed_root, job.set_name, job.tomoname, alignment_dir
        )
        if align_src is None:
            warnings.append(
                f"missing alignment dir {job.set_name}/{job.tomoname}/{alignment_dir}"
            )
            continue
        assets, asset_warns = collect_alignment_assets(
            align_src,
            job.tomoname,
            job.set_name,
            h4k2cys_etomo_dir=h4k2cys_etomo_dir,
            h4k2cys_tiltstack_root=h4k2cys_tiltstack_root,
        )
        warnings.extend(asset_warns)
        align_dest = dest_tomo / out_name
        for src, rel in assets:
            plan = plan_copy(src, align_dest / rel, f"alignment/{out_name}", force)
            if plan:
                plans.append(plan)

    return plans, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy processed + raw tomogram assets into CZII_data_upload layout."
    )
    parser.add_argument(
        "--tomocsv",
        type=Path,
        required=True,
        help="Tomogram CSV with tomoname,set,alignment_dir columns",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Processed tomograms root (default: {DEFAULT_PROCESSED_ROOT})",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Raw frames root (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"CZII upload destination (default: {DEFAULT_DEST_ROOT})",
    )
    parser.add_argument(
        "--h4k2cys-etomo-dir",
        type=Path,
        default=DEFAULT_H4K2CYS_ETOMO_DIR,
        help=(
            f"H4K2Cys Warp etomo dir with {{tomoname}}.xml "
            f"(default: {DEFAULT_H4K2CYS_ETOMO_DIR})"
        ),
    )
    parser.add_argument(
        "--h4k2cys-tiltstack-root",
        type=Path,
        default=DEFAULT_H4K2CYS_TILTSTACK_ROOT,
        help=(
            f"H4K2Cys Warp tiltstack root containing {{tomoname}}/{{tomoname}}.st "
            f"(default: {DEFAULT_H4K2CYS_TILTSTACK_ROOT})"
        ),
    )
    parser.add_argument(
        "--sets",
        nargs="*",
        default=None,
        help="Optional set name filter(s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional CSV path for missing-file / warning report",
    )
    parser.add_argument(
        "--write-aunp-xyz-stars",
        action="store_true",
        help=(
            "After copying, write sibling Relion STAR files containing only "
            "faCoordinateX/Y/Z (named *_xyz_only.star). Does not delete originals."
        ),
    )
    args = parser.parse_args(argv)

    if not args.tomocsv.is_file():
        print(f"ERROR: tomocsv not found: {args.tomocsv}", file=sys.stderr)
        return 1

    sets_filter = set(args.sets) if args.sets else None
    jobs = load_jobs(args.tomocsv, sets_filter)
    if not jobs:
        print("No tomogram rows to process.")
        return 1

    uses_h4k2cys = any(j.set_name == H4K2CYS_SET for j in jobs)
    print(
        f"Jobs: {len(jobs)} tomograms from {args.tomocsv}\n"
        f"  processed: {args.processed_root}\n"
        f"  raw:       {args.raw_root}\n"
        f"  dest:      {args.dest_root}\n"
        f"  dry_run:   {args.dry_run}\n"
        f"  xyz_stars: {args.write_aunp_xyz_stars}"
    )
    if uses_h4k2cys:
        print(
            f"  H4K2Cys etomo:     {args.h4k2cys_etomo_dir}\n"
            f"  H4K2Cys tiltstack: {args.h4k2cys_tiltstack_root}"
        )

    all_warnings: list[dict[str, str]] = []
    n_copied = 0
    n_planned = 0
    n_xyz = 0

    for job in jobs:
        print(f"\n=== {job.set_name}/{job.tomoname} ({', '.join(job.alignment_dirs)}) ===")
        plans, warnings = build_plans_for_job(
            job,
            processed_root=args.processed_root,
            raw_root=args.raw_root,
            dest_root=args.dest_root,
            force=args.force,
            h4k2cys_etomo_dir=args.h4k2cys_etomo_dir,
            h4k2cys_tiltstack_root=args.h4k2cys_tiltstack_root,
        )
        for w in warnings:
            print(f"  WARN: {w}")
            all_warnings.append(
                {
                    "set": job.set_name,
                    "tomoname": job.tomoname,
                    "warning": w,
                }
            )
        for plan in plans:
            n_planned += 1
            try:
                execute_copy(plan, dry_run=args.dry_run)
                n_copied += 1
            except Exception as exc:  # noqa: BLE001
                msg = f"copy failed {plan.src} -> {plan.dest}: {exc}"
                print(f"  ERROR: {msg}")
                all_warnings.append(
                    {
                        "set": job.set_name,
                        "tomoname": job.tomoname,
                        "warning": msg,
                    }
                )

        if args.write_aunp_xyz_stars:
            dest_tomo = args.dest_root / job.set_name / job.tomoname
            # Prefer stars just copied into annotations/; also pick up any already present
            star_targets: list[Path] = []
            for plan in plans:
                if (
                    plan.dest.suffix == ".star"
                    and "annotations" in plan.dest.parts
                    and not plan.dest.name.endswith(f"{XYZ_STAR_SUFFIX}.star")
                ):
                    star_targets.append(plan.dest)
            if not args.dry_run:
                star_targets.extend(annotation_stars_for_job(dest_tomo))
            # Unique, stable order
            seen: set[Path] = set()
            unique_stars: list[Path] = []
            for path in star_targets:
                key = path.resolve() if path.exists() else path
                if key in seen:
                    continue
                seen.add(key)
                unique_stars.append(path)

            for dest_star in unique_stars:
                src_for_xyz = dest_star
                if not src_for_xyz.exists():
                    for plan in plans:
                        if plan.dest == dest_star and plan.src.exists():
                            src_for_xyz = plan.src
                            break
                out = write_aunp_xyz_only_star(
                    src_for_xyz,
                    out_path=xyz_only_star_path(dest_star),
                    dry_run=args.dry_run,
                    force=args.force,
                )
                if out is not None:
                    n_xyz += 1

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["set", "tomoname", "warning"])
            writer.writeheader()
            writer.writerows(all_warnings)
        print(f"\nWrote report: {args.report} ({len(all_warnings)} warnings)")

    action = "planned" if args.dry_run else "copied"
    print(
        f"\nDone: {n_copied}/{n_planned} files {action}; "
        f"{n_xyz} xyz STAR(s); {len(all_warnings)} warnings"
    )
    return 0 if n_planned or n_xyz else 1


if __name__ == "__main__":
    raise SystemExit(main())
