#!/usr/bin/env python3
"""
Extract XYZ coordinates for specific AuNP clusters.

This script reads a CSV file that specifies tomogram names and cluster numbers,
then extracts the XYZ coordinates for those specific clusters and saves them
as text files.

Expected CSV format:
tomogram_name,cluster_number,set
20241030_AMmilled12-1_15,1,15F1
20241030_AMmilled12-1_15,2,15F1
20231017_EGmilled24-2_68,1,15F1
...

Output:
- Text files with XYZ coordinates for each specified cluster
- Files named: {tomogram_name}_cluster_{cluster_number}_coordinates.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import starfile
import sys
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

def generate_selected_clusters_pdf(successful_clusters, output_path):
    """
    Generate a PDF showing mini zonogram images for the selected clusters.
    
    Args:
        successful_clusters (list): List of dictionaries with cluster information
        output_path (Path): Output directory path
    """
    try:
        # Create PDF document
        pdf_path = output_path / "selected_clusters_summary.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Create custom style for tomogram names
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        # Group clusters by tomogram
        tomogram_clusters = {}
        for cluster_info in successful_clusters:
            tomogram_name = cluster_info['tomogram_name']
            if tomogram_name not in tomogram_clusters:
                tomogram_clusters[tomogram_name] = []
            tomogram_clusters[tomogram_name].append(cluster_info)
        
        # Process each tomogram
        for tomogram_name, clusters in tomogram_clusters.items():
            # Add tomogram name as title
            story.append(Paragraph(f"Tomogram: {tomogram_name}", title_style))
            story.append(Spacer(1, 10))
            
            # Find mini zonogram comparison files for this tomogram
            tomogram_path = clusters[0]['tomogram_path']
            activezonograms_dir = tomogram_path / "best_alignment" / "STT_results" / "activezonograms"
            
            if not activezonograms_dir.exists():
                story.append(Paragraph(f"Warning: No active zonogram directory found for {tomogram_name}", styles['Normal']))
                story.append(Spacer(1, 10))
                continue
            
            # Find mini zonogram comparison files for the selected clusters
            selected_cluster_files = []
            for cluster_info in clusters:
                cluster_number = cluster_info['cluster_number']
                pattern = f"*_mini_zonogram_cluster_{cluster_number}_comparison.png"
                matching_files = list(activezonograms_dir.glob(pattern))
                if matching_files:
                    selected_cluster_files.extend(matching_files)
            
            if not selected_cluster_files:
                story.append(Paragraph(f"No mini zonogram images found for selected clusters in {tomogram_name}", styles['Normal']))
                story.append(Spacer(1, 10))
                continue
            
            # Add mini zonograms in two columns
            for j in range(0, len(selected_cluster_files), 2):
                # Create a table-like layout for two columns
                from reportlab.platypus import Table, TableStyle
                
                row_data = []
                for k in range(2):
                    if j + k < len(selected_cluster_files):
                        mini_file = selected_cluster_files[j + k]
                        try:
                            # Get cluster number from filename
                            cluster_num = mini_file.stem.split('_cluster_')[1].split('_comparison')[0]
                            
                            # Add image (preserve aspect ratio but ensure it fits)
                            from PIL import Image as PILImage
                            pil_img = PILImage.open(str(mini_file))
                            orig_width, orig_height = pil_img.size
                            aspect_ratio = orig_width / orig_height
                            
                            # Calculate maximum dimensions that fit in two columns
                            max_width = 3.5 * inch
                            max_height = 300  # Leave some margin for mini zonograms
                            
                            # Calculate dimensions that preserve aspect ratio and fit within limits
                            if max_width / aspect_ratio <= max_height:
                                # Width is the limiting factor
                                final_width = max_width
                                final_height = max_width / aspect_ratio
                            else:
                                # Height is the limiting factor
                                final_height = max_height
                                final_width = max_height * aspect_ratio
                            
                            img = Image(str(mini_file), width=final_width, height=final_height)
                            
                            row_data.append([img])
                        except Exception as e:
                            print(f"  Error adding mini zonogram {mini_file}: {e}")
                            row_data.append([""])
                    else:
                        row_data.append([""])
                
                if any(cell != [""] for cell in row_data):
                    # Create table for this row
                    table = Table(row_data, colWidths=[3.5*inch, 3.5*inch])
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 10))
            
            story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        print(f"PDF saved to: {pdf_path}")
        return True
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def extract_cluster_coordinates(cluster_csv_path, data_dir, output_dir):
    """
    Extract XYZ coordinates for specific clusters based on CSV input.
    
    Args:
        cluster_csv_path (str): Path to CSV file with tomogram_name,cluster_number columns
        data_dir (str): Root directory containing tomogram data
        output_dir (str): Output directory for coordinate files
    """
    print(f"Reading cluster selection CSV: {cluster_csv_path}")
    
    # Read the cluster selection CSV
    try:
        cluster_df = pd.read_csv(cluster_csv_path)
        print(f"Found {len(cluster_df)} cluster specifications")
    except Exception as e:
        print(f"Error reading cluster CSV: {e}")
        return False
    
    # Check required columns
    required_columns = ['tomogram_name', 'cluster_number', 'set']
    missing_columns = [col for col in required_columns if col not in cluster_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in CSV: {missing_columns}")
        print(f"Expected columns: {required_columns}")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful_extractions = 0
    total_clusters = len(cluster_df)
    
    # Track successful extractions for PDF generation
    successful_clusters = []
    
    for idx, row in cluster_df.iterrows():
        tomogram_name = row['tomogram_name']
        cluster_number = row['cluster_number']
        set_name = row['set']
        
        print(f"\nProcessing {idx+1}/{total_clusters}: {tomogram_name}, Cluster {cluster_number}, Set {set_name}")
        
        # Construct tomogram path using the specified set
        tomogram_path = Path(data_dir) / set_name / 'TOP_TOMOS' / tomogram_name
        
        if not tomogram_path.exists():
            print(f"  Warning: Tomogram directory not found: {tomogram_path}")
            continue
        
        # Path to AuNP cluster data
        aunp_data_path = tomogram_path / "best_alignment" / "STT_results" / "aunps" / "aunp_clusters.star"
        
        if not aunp_data_path.exists():
            print(f"  Warning: AuNP cluster data not found at {aunp_data_path}")
            print(f"  Please run AuNP analysis on {tomogram_name} first")
            continue
        
        try:
            # Load AuNP cluster data
            aunp_df = starfile.read(aunp_data_path)
            
            # Filter for the specific cluster
            cluster_data = aunp_df[aunp_df['aunp_cluster'] == cluster_number]
            
            if len(cluster_data) == 0:
                print(f"  Warning: No AuNPs found for cluster {cluster_number} in {tomogram_name}")
                continue
            
            # Extract XYZ coordinates
            coordinates = cluster_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].values
            
            # Save to text file
            output_filename = f"{tomogram_name}_cluster_{cluster_number}_coordinates.txt"
            output_file = output_path / output_filename
            
            np.savetxt(output_file, coordinates, fmt='%.3f', 
                      header='X(nm)\tY(nm)\tZ(nm)', 
                      comments='# ')
            
            print(f"  Successfully extracted {len(coordinates)} coordinates to {output_filename}")
            successful_extractions += 1
            
            # Track for PDF generation
            successful_clusters.append({
                'tomogram_name': tomogram_name,
                'cluster_number': cluster_number,
                'set_name': set_name,
                'tomogram_path': tomogram_path,
                'aunp_count': len(coordinates)
            })
            
        except Exception as e:
            print(f"  Error processing {tomogram_name}, cluster {cluster_number}: {e}")
            continue
    
    print(f"\nExtraction completed: {successful_extractions}/{total_clusters} clusters successfully processed")
    print(f"Coordinate files saved to: {output_path}")
    
    # Generate PDF if we have successful extractions
    if successful_clusters:
        print("\nGenerating PDF summary of selected clusters...")
        pdf_success = generate_selected_clusters_pdf(successful_clusters, output_path)
        if pdf_success:
            print("PDF summary generated successfully!")
        else:
            print("Warning: Failed to generate PDF summary")
    
    return successful_extractions > 0

def main():
    parser = argparse.ArgumentParser(description='Extract XYZ coordinates for specific AuNP clusters')
    parser.add_argument('--cluster-csv', required=True, 
                       help='Path to CSV file with tomogram_name,cluster_number columns')
    parser.add_argument('--data-dir', required=True,
                       help='Root directory containing tomogram data')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for coordinate files')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.cluster_csv):
        print(f"Error: Cluster CSV file not found: {args.cluster_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run the extraction
    success = extract_cluster_coordinates(args.cluster_csv, args.data_dir, args.output_dir)
    
    if success:
        print("\nCluster coordinate extraction completed successfully!")
    else:
        print("\nCluster coordinate extraction completed with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
