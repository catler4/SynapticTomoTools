#!/usr/bin/env python3
"""
Generate summary PDF for each tomogram after analysis.
Includes: combined visualization, aunp cluster overlay, aunp clusters visualization, total aunp #, aunp cluster #, total vesicle #, and active zone-adjacent (<10 nm) vesicle #.
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

def get_stats(tomo_name, base_data_dir):
    # Find vesicle_results.json and aunp data for this tomogram
    vesicle_json = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/STT_results/vesicles/vesicle_results.json"))
    aunp_csv = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/STT_results/aunps/aunp_clusters.csv"))
    aunp_star_files = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/aunps/aunp_tm_BP_active_zone_*.star"))
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
                # Count vesicles with distance_to_az <= 10
                az_adj = 0
                for v in data.get('vesicles', []):
                    if 'distance_to_az' in v and v['distance_to_az'] <= 10:
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
    
    # AuNP stats - count total AuNPs from original star files
    total_aunps = 0
    if aunp_star_files:
        try:
            import starfile
            for star_file in aunp_star_files:
                # Skip the _all.star file if it exists
                if '_all.star' in str(star_file):
                    continue
                try:
                    star_data = starfile.read(star_file)
                    if 'particles' in star_data:
                        total_aunps += len(star_data['particles'])
                    elif 'data' in star_data:
                        total_aunps += len(star_data['data'])
                except Exception as e:
                    print(f"Warning: Error reading star file {star_file}: {e}")
            stats['total_aunps'] = total_aunps
        except Exception as e:
            print(f"Warning: Error processing AuNP star files for {tomo_name}: {e}")
    
    # AuNP cluster stats
    if aunp_csv:
        try:
            with open(aunp_csv[0], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                clusters = 0
                for row in reader:
                    clusters += 1
                stats['aunp_clusters'] = clusters
        except UnicodeDecodeError as e:
            print(f"Warning: Unicode decode error reading AuNP cluster results for {tomo_name}: {e}")
            print(f"File: {aunp_csv[0]}")
            print(f"File analysis: {check_file_corruption(aunp_csv[0])}")
        except Exception as e:
            print(f"Warning: Error reading AuNP cluster results for {tomo_name}: {e}")
            print(f"File: {aunp_csv[0]}")
            print(f"File analysis: {check_file_corruption(aunp_csv[0])}")
    return stats

def get_active_zonogram_images(tomo_name, base_data_dir):
    # Find the active_zonogram folder for this tomogram
    az_dir_candidates = list(Path(base_data_dir).glob(f"**/{tomo_name}/best_alignment/active_zonograms"))
    if not az_dir_candidates:
        return []
    az_dir = az_dir_candidates[0]
    # Find all *_position.png and *_selected_aunps.png pairs
    position_imgs = sorted(az_dir.glob('active_zonogram_*_position.png'))
    selected_imgs = sorted(az_dir.glob('active_zonogram_*_selected_aunps.png'))
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
    with open(tomocsv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tomo_set_map[row['tomoname']] = row['set']
            # Handle aunp_active_zones column (may be missing)
            az = (row.get('aunp_active_zones') or '').strip()
            if az:
                tomo_az_map[row['tomoname']] = az
            else:
                tomo_az_map[row['tomoname']] = 'All'
    return tomo_set_map, tomo_az_map

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

def generate_pdf_for_tomogram(tomo_name, vis_dir, base_data_dir, output_dir, tomo_set_map=None, tomo_az_map=None):
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
    stats = get_stats(tomo_name, base_data_dir)
    c.setFont("Helvetica", 12)
    c.drawString(margin, info_y, f"Total vesicles: {stats['total_vesicles']}")
    info_y -= 18
    c.drawString(margin, info_y, f"Active zone-adjacent vesicles (<10 nm): {stats['az_adjacent_vesicles']}")
    info_y -= 18
    c.drawString(margin, info_y, f"Total AuNPs: {stats['total_aunps']}")
    info_y -= 18
    c.drawString(margin, info_y, f"AuNP clusters: {stats['aunp_clusters']}")
    # After the box, set y to the bottom of the box
    y -= info_box_height
    # Add a consistent gap before the first figure
    y -= 16
    # --- Add active zonogram images next ---
    az_pairs = get_active_zonogram_images(tomo_name, base_data_dir)
    if az_pairs:
        # Place the first pair on the first page, stacked, using as much vertical space as possible
        pos_img, sel_img = az_pairs[0]
        gap = 28
        available_height = y - margin  # space left on first page
        img_height = (available_height - gap) // 2
        img_width = width - 2*margin
        # Position image 1
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, f"Active Zonogram 0 - Position")
        y -= 6
        used_height1 = add_image(c, pos_img, margin, y-img_height, img_width, img_height)
        y -= used_height1
        y -= gap
        # Position image 2
        if sel_img:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, y, f"Active Zonogram 0 - Selected AuNPs")
            y -= 6
            used_height2 = add_image(c, sel_img, margin, y-img_height, img_width, img_height)
            y -= used_height2
            y -= 12
        else:
            c.setFont("Helvetica", 12)
            c.drawString(margin, y, "[Selected AuNPs image not found]")
            y -= 20
        # Remove the first pair from az_pairs
        az_pairs = az_pairs[1:]
    # The rest as before
    for i, (pos_img, sel_img) in enumerate(az_pairs, start=1):
        if y < (2*400 + 40):
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, f"Active Zonogram {i} - Position")
        y -= 6
        used_height = add_image(c, pos_img, margin, y-400, width-2*margin, 400)
        y -= used_height
        y -= 4
        if sel_img:
            y -= 16
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, y, f"Active Zonogram {i} - Selected AuNPs")
            y -= 6
            used_height = add_image(c, sel_img, margin, y-400, width-2*margin, 400)
            y -= used_height
            y -= 4
    # --- End active zonogram images ---
    # Add images
    img_types = [
        ("Analysis Summary", f"{tomo_name}_combined.png"),
        ("AuNP Clusters Overlay", f"{tomo_name}_combined_aunpclusters.png"),
        ("AuNP Clusters 2D Projection", f"{tomo_name}_aunpclusters.png"),
    ]
    # Custom layout: first two side-by-side, third below
    img_paths = [vis_dir / fname for _, fname in img_types]
    img_labels = [label for label, _ in img_types]
    # Check if both side-by-side images exist
    if img_paths[0].exists() and img_paths[1].exists():
        # Calculate available width and height
        gap = 16
        side_width = (width - 2*margin - gap) // 2
        max_height = 320
        # Estimate needed height for all three images and titles
        needed_height = max_height + 16 + max_height + 20  # side-by-side row + gap + below row
        if img_paths[2].exists():
            needed_height += max_height + 22  # space for third image and its title
        if y < needed_height:
            c.showPage()
            y = height - margin
        # Titles above each image
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, img_labels[0])
        c.drawString(margin + side_width + gap, y, img_labels[1])
        y -= 16
        # Draw both images side by side
        nh1 = add_image(c, img_paths[0], margin, y-max_height, side_width, max_height)
        nh2 = add_image(c, img_paths[1], margin + side_width + gap, y-max_height, side_width, max_height)
        used_height = max(nh1, nh2)
        y -= used_height
        y -= 20
        # Third image below, if it exists
        if img_paths[2].exists():
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, y, img_labels[2])
            y -= 10
            nh3 = add_image(c, img_paths[2], margin, y-max_height, width-2*margin, max_height)
            y -= nh3
            y -= 12
    else:
        # Fallback: show images vertically as before
        for label, fname in img_types:
            img_path = vis_dir / fname
            if img_path.exists():
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, y, label)
                y -= 6
                used_height = add_image(c, img_path, margin, y-400, width-2*margin, 400)
                y -= used_height
                y -= 4
            else:
                c.setFont("Helvetica", 12)
                c.drawString(margin, y, f"[Image not found: {fname}]")
                y -= 20
    c.save()
    print(f"Saved PDF: {output_dir / f'{tomo_name}_summary.pdf'}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate tomogram summary PDFs')
    parser.add_argument('--vis-dir', default='results/visualizations', help='Directory with visualization images')
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
    tomo_set_map, tomo_az_map = load_tomo_set_map(args.tomocsv)
    
    # Get tomograms in CSV order
    csv_tomograms = list(tomo_set_map.keys())
    
    # Find all combined images as tomogram anchors
    allowed_tomos = set(tomo_set_map.keys())
    combined_imgs = [img for img in vis_dir.glob('*_combined.png') if img.name.replace('_combined.png', '') in allowed_tomos]
    
    # Create a mapping from tomogram name to image path
    tomo_to_img = {img.name.replace('_combined.png', ''): img for img in combined_imgs}
    
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
            print(f"Generating PDF for {tomo_name} (CSV order, position {i+1}/{len(csv_tomograms)})")
            generate_pdf_for_tomogram(tomo_name, vis_dir, base_data_dir, output_dir, tomo_set_map, tomo_az_map)
            pdf_path = output_dir / f"{tomo_name}_summary.pdf"
            if pdf_path.exists():
                pdf_paths.append(str(pdf_path))
        else:
            print(f"Skipping {tomo_name} - no combined image found")
    
    # For the final combined PDF, we need to include all tomograms from the original CSV
    # So we need to check for existing PDFs for all tomograms, not just the ones we just generated
    all_pdf_paths = []
    print(f"\nCollecting all available PDFs for combined document...")
    for tomo_name in csv_tomograms:
        pdf_path = output_dir / f"{tomo_name}_summary.pdf"
        if pdf_path.exists():
            all_pdf_paths.append(str(pdf_path))
            print(f"  ✓ Found PDF for {tomo_name}")
        else:
            print(f"  ✗ No PDF found for {tomo_name} - will be skipped in combined PDF")
    
    print(f"\nFound {len(all_pdf_paths)} PDFs to combine")
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
        for png in summary_pngs:
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
        merged_pdf_path = output_dir / "all_tomograms_summary.pdf"
        merger.write(str(merged_pdf_path))
        merger.close()
        print(f"Combined {len(all_pdf_paths)} PDFs into: {merged_pdf_path}")
        print(f"Note: Combined PDF includes all available tomogram summaries from the original CSV")
if __name__ == "__main__":
    main() 