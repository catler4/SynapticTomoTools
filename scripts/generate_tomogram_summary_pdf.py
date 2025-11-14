#!/usr/bin/env python3
"""
Generate summary PDF for each tomogram after analysis.
Includes: combined visualization, aunp cluster overlay, aunp clusters visualization, total aunp #, aunp cluster #, total vesicle #, and active zone-adjacent (<20 nm) vesicle #.
"""
import os
import glob
import json
import csv
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from PyPDF2 import PdfMerger
import re
from reportlab.lib.colors import HexColor

def get_tomo_name_from_image(image_path):
    # Assumes image name format: <tomo_name>_combined.png, etc.
    return '_'.join(Path(image_path).name.split('_')[:-1])

def check_file_corruption(file_path):
    """Check if a file appears to be corrupted by examining its first few bytes."""
    try:
        with open(file_path, 'rb') as f:
            # Read first 100 bytes to check for common corruption indicators
            header = f.read(100)
            
            # Check for gzip magic number
            if header.startswith(b'\x1f\x8b'):
                return "File appears to be gzipped (compressed)"
            
            # Check for common binary file indicators
            if b'\x00' in header[:20]:
                return "File contains null bytes (may be binary)"
            
            # Check for UTF-8 BOM
            if header.startswith(b'\xef\xbb\xbf'):
                return "File has UTF-8 BOM"
            
            # Check for other compression formats
            if header.startswith(b'PK'):
                return "File appears to be a ZIP archive"
            
            return "File appears to be text-based"
            
    except Exception as e:
        return f"Error reading file: {e}"

def get_stats(tomo_name, base_data_dir, selected_az_indices=None):
    # Find vesicle_results.json and aunp data for this tomogram
    vesicle_json = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/STT_results/vesicles/vesicle_results.json"))
    aunp_star = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/STT_results/aunps/aunp_clusters.star"))
    
    stats = {
        'total_vesicles': 'N/A',
        'az_adjacent_vesicles': 'N/A',
        'total_aunps': 'N/A',
        'aunp_clusters': 'N/A',
    }
    # Vesicle stats
    if vesicle_json:
        try:
            with open(vesicle_json[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats['total_vesicles'] = data.get('summary', {}).get('total_vesicles', 'N/A')
                # Count vesicles with distance_to_az <= 20
                az_adj = 0
                for v in data.get('vesicles', []):
                    if 'distance_to_az' in v and v['distance_to_az'] <= 20:
                        az_adj += 1
                stats['az_adjacent_vesicles'] = az_adj
        except UnicodeDecodeError as e:
            print(f"Warning: Unicode decode error reading vesicle results for {tomo_name}: {e}")
            print(f"File: {vesicle_json[0]}")
            print(f"File analysis: {check_file_corruption(vesicle_json[0])}")
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error reading vesicle results for {tomo_name}: {e}")
            print(f"File: {vesicle_json[0]}")
            print(f"File analysis: {check_file_corruption(vesicle_json[0])}")
        except Exception as e:
            print(f"Warning: Error reading vesicle results for {tomo_name}: {e}")
            print(f"File: {vesicle_json[0]}")
            print(f"File analysis: {check_file_corruption(vesicle_json[0])}")
    

    
    # AuNP stats - use the .star file for both total count and cluster count
    if aunp_star:
        try:
            import starfile
            df = starfile.read(aunp_star[0])
            
            # Filter by selected active zones if specified
            if selected_az_indices is not None:
                df = df[df['active_zone'].isin(selected_az_indices)]
            
            # Total AuNPs: count all AuNPs (including noise cluster -1)
            stats['total_aunps'] = len(df)
            
            # AuNP clusters: count unique valid clusters (excluding noise cluster -1)
            valid_df = df[(df['aunp_cluster'] != -1) & (df['aunp_cluster'].notna())]
            unique_clusters = valid_df['aunp_cluster'].unique()
            stats['aunp_clusters'] = len(unique_clusters)
                
        except Exception as e:
            print(f"Warning: Error reading AuNP cluster results for {tomo_name}: {e}")
            print(f"File: {aunp_star[0]}")
            print(f"File analysis: {check_file_corruption(aunp_star[0])}")
    return stats

def get_active_zonogram_images(tomo_name, base_data_dir, selected_az_indices=None):
    # Find the active_zonogram folder for this tomogram
    az_dir_candidates = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/active_zonograms"))
    if not az_dir_candidates:
        return []
    az_dir = az_dir_candidates[0]
    
    # Find all *_position.png and *_selected_aunps.png pairs
    position_imgs = sorted(az_dir.glob('active_zonogram_*_position.png'))
    selected_imgs = sorted(az_dir.glob('active_zonogram_*_selected_aunps.png'))
    
    # Filter by selected active zone indices if provided
    if selected_az_indices is not None:
        # Convert to set for faster lookup
        selected_indices = set(selected_az_indices)
        # Filter position images to only include selected indices
        filtered_position_imgs = []
        for pos_img in position_imgs:
            # Extract index from filename like: active_zonogram_0_position.png
            m = re.search(r'active_zonogram_(\d+)_position.png', pos_img.name)
            if m:
                idx = int(m.group(1))
                if idx in selected_indices:
                    filtered_position_imgs.append(pos_img)
        position_imgs = filtered_position_imgs
    
    # Pair by index (assuming same numbering)
    pairs = []
    for pos_img in position_imgs:
        # Extract index
        m = re.search(r'active_zonogram_(\d+)_position.png', pos_img.name)
        if not m:
            continue
        idx = m.group(1)
        # Find corresponding selected_aunps image
        sel_img = az_dir / f"active_zonogram_{idx}_selected_aunps.png"
        if sel_img.exists():
            pairs.append((pos_img, sel_img))
        else:
            pairs.append((pos_img, None))
    return pairs

def load_tomo_set_map(tomocsv_path):
    import csv
    tomo_set_map = {}
    tomo_az_map = {}
    tomo_az_indices_map = {}
    with open(tomocsv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tomo_set_map[row['tomoname']] = row['set']
            # Handle aunp_active_zones column (may be missing)
            az = (row.get('aunp_active_zones') or '').strip()
            if az:
                tomo_az_map[row['tomoname']] = az
                # Parse active zone indices (handle floats like "2.0")
                try:
                    az_indices = []
                    for x in az.split(','):
                        x = x.strip()
                        if x.isdigit():
                            az_indices.append(int(x))
                        elif x.replace(".", "").isdigit():  # Handle floats like "2.0"
                            az_indices.append(int(float(x)))
                    tomo_az_indices_map[row['tomoname']] = az_indices
                except ValueError:
                    print(f"Warning: Could not parse active zone indices for {row['tomoname']}: {az}")
                    tomo_az_indices_map[row['tomoname']] = None
            else:
                tomo_az_map[row['tomoname']] = 'All'
                tomo_az_indices_map[row['tomoname']] = None
    return tomo_set_map, tomo_az_map, tomo_az_indices_map

def add_image(c, img_path, x, y, max_width, max_height):
    if img_path is None:
        print(f"Warning: Attempted to add None image path")
        return 0
    
    if not img_path.exists():
        print(f"Warning: Image file does not exist: {img_path}")
        return 0
    
    try:
        img = Image.open(img_path)
        iw, ih = img.size
        width = min(iw, max_width)
        height = min(ih, max_height)
        scale = min(width / iw, height / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img_io = ImageReader(img)
        c.drawImage(img_io, x, y + (max_height - nh), width=nw, height=nh)
        return nh  # Return the actual height used
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return 0

def generate_pdf_for_tomogram(tomo_name, vis_dir, base_data_dir, output_dir, tomo_set_map=None, tomo_az_map=None, tomo_az_indices_map=None):
    c = canvas.Canvas(str(output_dir / f"{tomo_name}_summary.pdf"), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    # --- Summary info first ---
    set_name = None
    az_info = None
    if tomo_set_map is not None:
        set_name = tomo_set_map.get(tomo_name, None)
    if tomo_az_map is not None:
        az_info = tomo_az_map.get(tomo_name, None)
    # Draw shaded box for tomogram title
    title_box_height = 40
    title_box_color = HexColor('#cccccc')  # medium grey
    c.setFillColor(title_box_color)
    c.setStrokeColor(title_box_color)
    c.rect(margin-8, y-title_box_height, width-2*margin+16, title_box_height, fill=1, stroke=0)
    c.setFillColor('black')
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y - 28, f"Tomogram Summary: {tomo_name}")
    y -= title_box_height
    # Draw lighter shaded box for set, az, and summary info
    info_box_height = 18+16+18*4+12
    info_box_color = HexColor('#eeeeee')  # light grey
    c.setFillColor(info_box_color)
    c.setStrokeColor(info_box_color)
    c.rect(margin-8, y-info_box_height, width-2*margin+16, info_box_height, fill=1, stroke=0)
    c.setFillColor('black')
    # Draw all info text inside the box using a fixed offset from the top of the box
    info_y = y - 14  # Start a little below the top of the box
    if set_name:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, info_y, f"Set: {set_name}")
        info_y -= 18
    if az_info:
        c.setFont("Helvetica", 12)
        c.drawString(margin, info_y, f"Active zones included: {az_info}")
        info_y -= 16
    # Get selected active zone indices for this tomogram
    selected_az_indices = None
    if tomo_az_indices_map is not None:
        selected_az_indices = tomo_az_indices_map.get(tomo_name, None)
    
    stats = get_stats(tomo_name, base_data_dir, selected_az_indices)
    c.setFont("Helvetica", 12)
    c.drawString(margin, info_y, f"Total vesicles: {stats['total_vesicles']}")
    info_y -= 18
    c.drawString(margin, info_y, f"Active zone-adjacent vesicles (<20 nm): {stats['az_adjacent_vesicles']}")
    info_y -= 18
    c.drawString(margin, info_y, f"Total AuNPs: {stats['total_aunps']}")
    info_y -= 18
    c.drawString(margin, info_y, f"AuNP clusters: {stats['aunp_clusters']}")
    # After the box, set y to the bottom of the box
    y -= info_box_height
    # Add a consistent gap before the first figure
    y -= 16
    # Get selected active zone indices for this tomogram
    selected_az_indices = None
    if tomo_az_indices_map is not None:
        selected_az_indices = tomo_az_indices_map.get(tomo_name, None)
    # Add images - create one set of all visualizations per active zone
    img_types = []
    if selected_az_indices is not None and len(selected_az_indices) > 0:
        # Add all visualizations for each active zone
        for az_idx in selected_az_indices:
            # Add active zonogram images for this active zone
            img_types.append((f"Active Zonogram Position (AZ {az_idx})", f"active_zonogram_{az_idx}_position.png"))
            img_types.append((f"Active Zonogram Selected AuNPs (AZ {az_idx})", f"active_zonogram_{az_idx}_selected_aunps.png"))
            # Add main visualizations for this active zone
            img_types.append((f"Analysis Summary (AZ {az_idx})", f"{tomo_name}_combined_az{az_idx}.png"))
            img_types.append((f"AuNP Clusters Overlay (AZ {az_idx})", f"{tomo_name}_combined_aunpclusters_az{az_idx}.png"))
            # Add cluster image for this active zone - need to find the actual zone name
            # Look for the actual active zonogram cluster file
            az_dir = vis_dir.parent / "active_zonograms"
            cluster_pattern = f"{tomo_name}_active_zonogram_*_selected_aunps_by_cluster_az{az_idx}.png"
            print(f"Looking for cluster files with pattern: {cluster_pattern}")
            print(f"In directory: {az_dir}")
            cluster_files = list(az_dir.glob(cluster_pattern))
            print(f"Found cluster files: {[f.name for f in cluster_files]}")
            if cluster_files:
                actual_filename = cluster_files[0].name
                print(f"Adding cluster image: {actual_filename}")
                img_types.append((f"Active Zone AuNP Clusters (AZ {az_idx})", actual_filename))
            else:
                # Fallback to default pattern
                fallback_filename = f"{tomo_name}_active_zonogram_active_zone_pre1_post1_selected_aunps_by_cluster_az{az_idx}.png"
                print(f"No cluster files found, using fallback: {fallback_filename}")
                img_types.append((f"Active Zone AuNP Clusters (AZ {az_idx})", fallback_filename))
    else:
        # No active zones specified - use az0 as default
        img_types = [
            ("Active Zonogram Position (AZ 0)", f"active_zonogram_0_position.png"),
            ("Active Zonogram Selected AuNPs (AZ 0)", f"active_zonogram_0_selected_aunps.png"),
            ("Analysis Summary (AZ 0)", f"{tomo_name}_combined_az0.png"),
            ("AuNP Clusters Overlay (AZ 0)", f"{tomo_name}_combined_aunpclusters_az0.png"),
            ("Active Zone AuNP Clusters (AZ 0)", f"{tomo_name}_active_zonogram_active_zone_pre1_post1_selected_aunps_by_cluster_az0.png"),
        ]
    # Group images by active zone and render each group
    img_paths = []
    for _, fname in img_types:
        print(f"Processing image: {fname}")
        if "active_zonogram" in fname and "active_zone_pre1_post1" not in fname and "selected_aunps_by_cluster" not in fname:
            # Active zonogram position/selected images are in the tomogram's active_zonograms directory
            # Find the actual tomogram directory (it might be in a subdirectory)
            tomo_dirs = list(base_data_dir.glob(f"**/{tomo_name}"))
            if tomo_dirs:
                az_dir = tomo_dirs[0] / "best_alignment" / "active_zonograms"
                img_path = az_dir / fname
                print(f"  Using tomogram directory: {img_path}")
            else:
                # Fallback to direct path
                az_dir = base_data_dir / f"{tomo_name}" / "best_alignment" / "active_zonograms"
                img_path = az_dir / fname
                print(f"  Using fallback path: {img_path}")
            img_paths.append(img_path)
        elif "active_zonogram" in fname:
            # Active zonogram cluster images are in the results active_zonograms directory
            # Also check the tomogram's STT_results/visualizations/active_zonograms directory
            az_dir = vis_dir.parent / "active_zonograms"
            img_path = az_dir / fname
            print(f"  Using results active_zonograms directory: {img_path}")
            print(f"  File exists: {img_path.exists()}")
            
            # Also check tomogram directory
            tomo_dirs = list(base_data_dir.glob(f"**/{tomo_name}"))
            if tomo_dirs:
                tomo_az_dir = tomo_dirs[0] / "best_alignment" / "STT_results" / "visualizations" / "active_zonograms"
                tomo_img_path = tomo_az_dir / fname
                print(f"  Also checking tomogram directory: {tomo_img_path}")
                print(f"  File exists in tomogram: {tomo_img_path.exists()}")
            
            img_paths.append(img_path)
        else:
            # Regular visualization images are in the aunps_and_vesicles subdirectory
            img_path = vis_dir / fname
            print(f"  Using aunps_and_vesicles directory: {img_path}")
            img_paths.append(img_path)
    img_labels = [label for label, _ in img_types]
    
    # Group images by active zone
    az_groups = {}
    for i, (label, path) in enumerate(zip(img_labels, img_paths)):
        # Extract active zone number from label
        if "(AZ " in label:
            az_num = label.split("(AZ ")[1].split(")")[0]
            if az_num not in az_groups:
                az_groups[az_num] = []
            az_groups[az_num].append((label, path))
    
    # Render each active zone group across 2 pages
    for i, az_num in enumerate(sorted(az_groups.keys())):
        az_images = az_groups[az_num]
        
        # Separate different types of images
        zonogram_imgs = []
        main_vis = []
        cluster_img = None
        for label, path in az_images:
            if "Active Zonogram Position" in label or "Active Zonogram Selected AuNPs" in label:
                zonogram_imgs.append((label, path))
            elif "Active Zone AuNP Clusters" in label:
                cluster_img = (label, path)
            else:
                main_vis.append((label, path))
        
        # Check if we have the required main visualization images
        if len(main_vis) >= 2 and main_vis[0][1].exists() and main_vis[1][1].exists():
            # Calculate available width and height
            gap = 16
            side_width = (width - 2*margin - gap) // 2
            max_height = 320
            
            # PAGE 1: Active zonogram images (stacked vertically)
            # For the first active zone, put zonogram images on the same page as header
            # For subsequent active zones, start a new page
            if zonogram_imgs:
                if i > 0:  # Not the first active zone
                    # Check if we need a new page
                    needed_height = (max_height + 22) * len(zonogram_imgs)
                    if y < needed_height:
                        c.showPage()
                        y = height - margin
                
                # Draw active zonogram images (stacked vertically)
                for label, path in zonogram_imgs:
                    if path.exists():
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(margin, y, label)
                        y -= 10
                        nh = add_image(c, path, margin, y-max_height, width-2*margin, max_height)
                        y -= nh
                        y -= 12
            
            # PAGE 2: Main visualizations (side-by-side) and cluster image
            c.showPage()
            y = height - margin
            
            # Titles above each main visualization image
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, y, main_vis[0][0])
            c.drawString(margin + side_width + gap, y, main_vis[1][0])
            y -= 16
            
            # Draw both main visualization images side by side
            nh1 = add_image(c, main_vis[0][1], margin, y-max_height, side_width, max_height)
            nh2 = add_image(c, main_vis[1][1], margin + side_width + gap, y-max_height, side_width, max_height)
            used_height = max(nh1, nh2)
            y -= used_height
            y -= 20
            
            # Draw cluster image below if it exists
            if cluster_img and cluster_img[1].exists():
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, y, cluster_img[0])
                y -= 10
                nh = add_image(c, cluster_img[1], margin, y-max_height, width-2*margin, max_height)
                y -= nh
                y -= 12
        else:
            # Required images not found for this active zone - fail with error
            missing_files = []
            for label, path in az_images:
                if not path.exists():
                    missing_files.append(str(path))
            
            if missing_files:
                error_msg = f"Error: Required image files not found for {tomo_name} Active Zone {az_num}:\n" + "\n".join(missing_files)
                print(error_msg)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, y, f"ERROR: Missing required images for {tomo_name} Active Zone {az_num}")
                y -= 30
                c.setFont("Helvetica", 12)
                for missing_file in missing_files:
                    c.drawString(margin, y, f"Missing: {missing_file}")
                    y -= 20
                c.save()
                return
    c.save()
    print("✅")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate tomogram summary PDFs')
    parser.add_argument('--vis-dir', default='results/visualizations/aunps_and_vesicles', help='Directory with visualization images')
    parser.add_argument('--data-dir', default='data', help='Base data directory for tomogram stats')
    parser.add_argument('--output-dir', default='results/summary_pdfs', help='Output directory for PDFs')
    parser.add_argument('--tomocsv', default='data/tomograms.csv', help='CSV file mapping tomogram names to sets')
    parser.add_argument('--start-from', default=None, help='Start PDF generation from this tomogram (still includes all tomograms in final PDF)')
    args = parser.parse_args()
    vis_dir = Path(args.vis_dir)
    base_data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load tomogram set and active zone mapping
    tomo_set_map, tomo_az_map, tomo_az_indices_map = load_tomo_set_map(args.tomocsv)
    
    # Get tomograms in CSV order
    csv_tomograms = list(tomo_set_map.keys())
    
    # Find all combined images as tomogram anchors
    allowed_tomos = set(tomo_set_map.keys())
    # vis_dir is already the aunps_and_vesicles directory, don't append it again
    aunps_vesicles_dir = vis_dir
    
    # Create a mapping from tomogram name to image path
    tomo_to_img = {}
    
    # Looking for combined images
    
    if aunps_vesicles_dir.exists():
        # Look for combined images with the correct active zone suffix for each tomogram
        for tomo_name in allowed_tomos:
            # Get the active zone indices for this tomogram
            az_indices = tomo_az_indices_map.get(tomo_name, [0])  # Default to [0] if not specified
            if az_indices is None:
                az_indices = [0]
            
            # Processing tomogram
            
            # Use the first active zone index for the combined image
            az_suffix = az_indices[0] if len(az_indices) > 0 else 0
            combined_img_path = aunps_vesicles_dir / f"{tomo_name}_combined_az{az_suffix}.png"
            
            # Looking for combined image
            if combined_img_path.exists():
                tomo_to_img[tomo_name] = combined_img_path
                # Found combined image
            else:
                # Combined image not found
                pass
    else:
        print(f"Directory {aunps_vesicles_dir} does not exist, using fallback")
        # Fallback to old location
        combined_imgs = [img for img in vis_dir.glob('*_combined.png') if img.name.replace('_combined.png', '') in allowed_tomos]
        tomo_to_img = {img.name.replace('_combined.png', ''): img for img in combined_imgs}
    
    # Final tomogram mapping completed
    
    pdf_paths = []
    # Process tomograms in CSV order, starting from specified tomogram if given
    start_index = 0
    if args.start_from:
        try:
            start_index = csv_tomograms.index(args.start_from)
            print(f"Starting PDF generation from tomogram: {args.start_from} (index {start_index})")
        except ValueError:
            print(f"Warning: Starting tomogram '{args.start_from}' not found in CSV, starting from beginning")
            start_index = 0
    
    # Process tomograms starting from the specified index
    for i, tomo_name in enumerate(csv_tomograms[start_index:], start=start_index):
        if tomo_name in tomo_to_img:
            print(f"[{i+1}/{len(csv_tomograms)}] Generating PDF for {tomo_name}...", end=" ", flush=True)
            generate_pdf_for_tomogram(tomo_name, vis_dir, base_data_dir, output_dir, tomo_set_map, tomo_az_map, tomo_az_indices_map)
            pdf_path = output_dir / f"{tomo_name}_summary.pdf"
            if pdf_path.exists():
                pdf_paths.append(str(pdf_path))
        else:
            print(f"Skipping {tomo_name} - no combined image found")
    
    # For the final combined PDF, we need to include all tomograms from the original CSV
    # So we need to check for existing PDFs for all tomograms, not just the ones we just generated
    all_pdf_paths = []
    print(f"\nCollecting PDFs for combined document...")
    for tomo_name in csv_tomograms:
        pdf_path = output_dir / f"{tomo_name}_summary.pdf"
        if pdf_path.exists():
            all_pdf_paths.append(str(pdf_path))
            # Found PDF
        else:
            # No PDF found - will be skipped
            pass
    
    # Found PDFs to combine
    # Combine all PDFs into a single document
    if all_pdf_paths:
        merger = PdfMerger()
        for pdf in all_pdf_paths:  # Use CSV order, not alphabetical order
            merger.append(pdf)
        # Add summary figures at the end
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        import glob
        import tempfile
        # Title page for summary figures
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_title:
            c = canvas.Canvas(tmp_title.name, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString(width/2, height/2 + 20, "Analysis Summary Figures by Set")
            c.setFont("Helvetica", 14)
            c.drawCentredString(width/2, height/2 - 10, "(Generated from all analyzed tomograms)")
            c.save()
            merger.append(tmp_title.name)
        # Add each summary PNG as a page
        summary_dir = output_dir
        summary_pngs = sorted(glob.glob(str(summary_dir / '*_by_set.png')))
        print(f"Found {len(summary_pngs)} summary figure files to add to PDF")
        successful_count = 0
        for png in summary_pngs:
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
                    c = canvas.Canvas(tmp_pdf.name, pagesize=letter)
                    width, height = letter
                    # Title from filename
                    title = os.path.basename(png).replace('_by_set.png', '').replace('_', ' ').title()
                    c.setFont("Helvetica-Bold", 16)
                    c.drawCentredString(width/2, height-40, title)
                    # Add image
                    img = Image.open(png)
                    iw, ih = img.size
                    max_width = width - 80
                    max_height = height - 120
                    scale = min(max_width/iw, max_height/ih, 1.0)
                    nw, nh = int(iw*scale), int(ih*scale)
                    x = (width - nw) // 2
                    y = (height - nh) // 2 - 20
                    c.drawImage(ImageReader(img), x, y, width=nw, height=nh)
                    c.save()
                    merger.append(tmp_pdf.name)
                    successful_count += 1
                    print(f"  Added: {os.path.basename(png)}")
            except Exception as e:
                print(f"  Error adding {os.path.basename(png)}: {e}")
                import traceback
                traceback.print_exc()
                continue
        print(f"Successfully added {successful_count} of {len(summary_pngs)} summary figures to PDF")
        merged_pdf_path = output_dir / "all_tomograms_summary.pdf"
        merger.write(str(merged_pdf_path))
        merger.close()
        print(f"✓ Combined {len(all_pdf_paths)} PDFs into: {merged_pdf_path}")
if __name__ == "__main__":
    main() 