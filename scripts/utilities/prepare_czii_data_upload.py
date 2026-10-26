#!/usr/bin/env python3
"""
Assemble a CZII public-upload tree from processed tomograms + raw frames.

Reads a tomogram CSV (``tomoname``, ``set``, ``alignment_dir``) and copies:

  {dest}/{set}/{tomoname}/
    frames/                         # raw movies + mdocs + gain (once per tomo)
    {renamed_alignment}/
      {tomoname}.defocus
      {tomoname}.mrc                # motion-corrected tilt stack
      {tomoname}_full_rec_BP_3DCTF_BIN4.mrc
      {tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc
      annotations/
        AuNP pick STARs, MemBrain smooth seg, pre/post .glb

Alignment rename:
  best_alignment      -> IMOD_alignment
  liza_azN[_rerun]    -> alignment_cleftN

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

FRAME_SUFFIXES = (".tif", ".eer", ".mrc")
GAIN_NAME_RE = re.compile(r"^gain_reference\.", re.IGNORECASE)
LIZA_AZ_RE = re.compile(r"^liza_az(\d+)(?:_rerun)?$", re.IGNORECASE)
POSITION_RE = re.compile(r"(Position[_\-][\w\-]+)", re.IGNORECASE)
DIMER_SET = "15F1and5F11dimer"

# AuNP pick patterns (basename globs under aunps/)
AUNP_STAR_DEFAULT = "aunp_tm_BP_active_zone_*_manual_refined.star"
AUNP_STAR_MONOMER = "aunp_tm_BP_active_zone_*_manual_refined_monomer.star"
AUNP_STAR_DIMER = "aunp_tm_BP_active_zone_*_manual_refined_dimer.star"
# Also accept hyphen variant if present on disk
AUNP_STAR_MONOMER_ALT = "aunp_tm_BP_active_zone_*_manual_refined-monomer.star"
AUNP_STAR_DIMER_ALT = "aunp_tm_BP_active_zone_*_manual_refined-dimer.star"


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
        return "IMOD_alignment"
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
    """Return Position_… substring used to match EER/mdoc basenames, if any."""
    m = POSITION_RE.search(tomoname)
    return m.group(1) if m else None


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


def frame_matches_tomo(path: Path, tomoname: str, pos: str | None) -> bool:
    stem = path.name
    if tomoname in stem:
        return True
    if pos and pos in stem:
        return True
    # Also match without date prefix: strip leading YYYYMMDD_ if present
    m = re.match(r"^\d{8}_(.+)$", tomoname)
    if m and m.group(1) in stem:
        return True
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


def collect_from_directory(directory: Path, tomoname: str) -> tuple[list[Path], list[Path], list[Path]]:
    """Collect frames/mdocs/gain from a tilt-series directory (layout A or mixed)."""
    pos = position_token(tomoname)
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
                if frame_matches_tomo(path, tomoname, pos) or path.parent == directory:
                    seen.add(resolved)
                    mdocs.append(path)
                continue
            if is_frame_file(path) and frame_matches_tomo(path, tomoname, pos):
                seen.add(resolved)
                frames.append(path)

    # If directory itself is the series folder and contains unmatched frames that
    # all share the tomoname prefix, take all non-gain frames/mdocs in that folder.
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


def find_raw_assets(
    raw_root: Path, set_name: str, tomoname: str
) -> tuple[list[Path], list[Path], list[Path], list[str]]:
    """
    Locate raw frames + mdocs + gain for one tomogram.

    Supports:
    - Series folder with per-tilt .tif/.eer/.mrc (+ .mdoc)
    - Session folder with frames/ + mdoc/ (EER style)
    """
    warnings: list[str] = []
    set_root = raw_root / set_name
    if not set_root.is_dir():
        return [], [], [], [f"raw set root missing: {set_root}"]

    # 1) Directory named exactly as tomoname
    named = find_named_dirs(set_root, tomoname)
    frames: list[Path] = []
    mdocs: list[Path] = []
    gains: list[Path] = []

    for d in named:
        f, m, g = collect_from_directory(d, tomoname)
        frames.extend(f)
        mdocs.extend(m)
        gains.extend(g)

    # 2) If nothing found, search session frames/ folders by Position / name match
    if not frames:
        pos = position_token(tomoname)
        for dirpath, dirnames, _filenames in os_walk_limited(set_root, max_depth=3):
            p = Path(dirpath)
            if p.name != "frames":
                continue
            for path in p.iterdir():
                if path.is_file() and is_frame_file(path) and frame_matches_tomo(
                    path, tomoname, pos
                ):
                    frames.append(path)
            # sibling mdoc/
            mdoc_dir = p.parent / "mdoc"
            if mdoc_dir.is_dir():
                for path in mdoc_dir.iterdir():
                    if path.is_file() and is_mdoc_file(path) and frame_matches_tomo(
                        path, tomoname, pos
                    ):
                        mdocs.append(path)
            for gpath in p.parent.iterdir():
                if gpath.is_file() and is_gain_file(gpath):
                    gains.append(gpath)
            # Also per-frame mdocs next to eers (rare)
            for path in p.iterdir():
                if path.is_file() and is_mdoc_file(path) and frame_matches_tomo(
                    path, tomoname, pos
                ):
                    mdocs.append(path)

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


def collect_alignment_assets(
    align_path: Path, tomoname: str, set_name: str
) -> tuple[list[tuple[Path, str]], list[str]]:
    """
    Return (src, relative_dest_under_alignment_dir) pairs and warnings.

    relative_dest is e.g. ``tomoname.defocus`` or ``annotations/foo.star``.
    """
    warnings: list[str] = []
    items: list[tuple[Path, str]] = []

    required = [
        f"{tomoname}.defocus",
        f"{tomoname}.mrc",
        f"{tomoname}_full_rec_BP_3DCTF_BIN4.mrc",
        f"{tomoname}_full_rec_BP_3DCTF_BIN4_ddw.mrc",
    ]
    for name in required:
        src = align_path / name
        if src.exists():
            items.append((src, name))
        else:
            warnings.append(f"missing {src}")

    # Annotations
    ann_sources: list[Path] = []
    aunps_dir = align_path / "aunps"
    membrain_dir = align_path / "membrain"

    if set_name == DIMER_SET:
        patterns = [AUNP_STAR_MONOMER, AUNP_STAR_DIMER, AUNP_STAR_MONOMER_ALT, AUNP_STAR_DIMER_ALT]
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

    if membrain_dir.is_dir():
        smooth = sorted(membrain_dir.glob("*_smooth.mrc"))
        if smooth:
            ann_sources.extend(smooth)
        else:
            warnings.append(f"no *_smooth.mrc under {membrain_dir}")
    else:
        warnings.append(f"missing membrain/ under {align_path}")

    if set_name == DIMER_SET:
        if not any("monomer" in p.name for p in ann_sources if p.suffix == ".star"):
            warnings.append(f"no monomer STAR found under {aunps_dir}")
        if not any("dimer" in p.name for p in ann_sources if p.suffix == ".star"):
            warnings.append(f"no dimer STAR found under {aunps_dir}")
    else:
        if not any(p.suffix == ".star" for p in ann_sources):
            warnings.append(f"no manual_refined STAR found under {aunps_dir}")

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


def build_plans_for_job(
    job: TomogramJob,
    *,
    processed_root: Path,
    raw_root: Path,
    dest_root: Path,
    force: bool,
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
            align_src, job.tomoname, job.set_name
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
    args = parser.parse_args(argv)

    if not args.tomocsv.is_file():
        print(f"ERROR: tomocsv not found: {args.tomocsv}", file=sys.stderr)
        return 1

    sets_filter = set(args.sets) if args.sets else None
    jobs = load_jobs(args.tomocsv, sets_filter)
    if not jobs:
        print("No tomogram rows to process.")
        return 1

    print(
        f"Jobs: {len(jobs)} tomograms from {args.tomocsv}\n"
        f"  processed: {args.processed_root}\n"
        f"  raw:       {args.raw_root}\n"
        f"  dest:      {args.dest_root}\n"
        f"  dry_run:   {args.dry_run}"
    )

    all_warnings: list[dict[str, str]] = []
    n_copied = 0
    n_planned = 0

    for job in jobs:
        print(f"\n=== {job.set_name}/{job.tomoname} ({', '.join(job.alignment_dirs)}) ===")
        plans, warnings = build_plans_for_job(
            job,
            processed_root=args.processed_root,
            raw_root=args.raw_root,
            dest_root=args.dest_root,
            force=args.force,
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

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["set", "tomoname", "warning"])
            writer.writeheader()
            writer.writerows(all_warnings)
        print(f"\nWrote report: {args.report} ({len(all_warnings)} warnings)")

    action = "planned" if args.dry_run else "copied"
    print(f"\nDone: {n_copied}/{n_planned} files {action}; {len(all_warnings)} warnings")
    return 0 if n_planned else 1


if __name__ == "__main__":
    raise SystemExit(main())
