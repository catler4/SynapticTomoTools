import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class ResultsManager:
    """
    Manages storage and retrieval of analysis results for tomograms.
    Results are stored in JSON format for easy inspection and modification.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_file = self.results_dir / "analysis_results.json"
        self.results = self._load_existing_results()
    
    def _load_existing_results(self) -> Dict[str, Any]:
        """Load existing results from JSON file."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_results(self):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def store_tomogram_results(
        self,
        tomogram_name: str,
        analysis_type: str,
        results: Dict[str, Any],
        overwrite: bool = False,
        set_name: Optional[str] = None,
        alignment_dir: Optional[str] = None,
    ):
        """
        Store results for a specific tomogram and analysis type.
        
        Args:
            tomogram_name: Name of the tomogram
            analysis_type: Type of analysis (e.g., 'activezone', 'vesicles', 'aunps')
            results: Dictionary of results to store
            overwrite: Whether to overwrite existing results
            set_name: Name of the experimental set (optional)
            alignment_dir: Alignment directory name (optional)
        """
        if tomogram_name not in self.results:
            self.results[tomogram_name] = {}
        
        if analysis_type in self.results[tomogram_name] and not overwrite:
            print(f"Warning: Results for {tomogram_name} - {analysis_type} already exist. Use overwrite=True to replace.")
            return
        
        # Add metadata
        results_with_metadata = {
            'results': results,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'set_name': set_name,
            'alignment_dir': alignment_dir,
        }
        
        self.results[tomogram_name][analysis_type] = results_with_metadata
        self._save_results()
        print(f"Stored results for {tomogram_name} - {analysis_type}")
    
    def get_tomogram_results(self, tomogram_name: str, analysis_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve results for a tomogram.
        
        Args:
            tomogram_name: Name of the tomogram
            analysis_type: Specific analysis type to retrieve (None for all)
        
        Returns:
            Dictionary of results
        """
        if tomogram_name not in self.results:
            return {}
        
        if analysis_type:
            return self.results[tomogram_name].get(analysis_type, {})
        else:
            return self.results[tomogram_name]
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all stored results."""
        return self.results
    
    def _add_units_to_column_name(self, col_name: str) -> str:
        """
        Add units to column names based on their content.
        Returns the column name with appropriate unit suffix if applicable.
        """
        # Distance measurements (nm)
        if any(term in col_name.lower() for term in ['distance', 'cleft_width', 'diameter', 'radius', 'dimension']):
            if '_nm' not in col_name and '_um' not in col_name:
                return f"{col_name}_nm"
        
        # Area measurements (µm²)
        if any(term in col_name.lower() for term in ['area', 'surface']):
            if '_nm2' not in col_name and '_um2' not in col_name and '_nm²' not in col_name and '_um²' not in col_name:
                return f"{col_name}_um2"
        
        # Volume measurements (µm³) — skip dimensionless names like sphericity_volume
        if 'volume' in col_name.lower() and 'sphericity' not in col_name.lower():
            if '_nm3' not in col_name and '_um3' not in col_name and '_nm³' not in col_name and '_um³' not in col_name:
                return f"{col_name}_um3"
        
        # Density measurements
        if 'density' in col_name.lower() and 'aunp' in col_name.lower():
            if '_per_um2' not in col_name and '_per_um²' not in col_name:
                return f"{col_name}_per_um2"
        
        # Coordinates (nm)
        if any(term in col_name.lower() for term in ['coordinate', 'center', 'point', 'x', 'y', 'z']):
            if col_name.lower() in ['x', 'y', 'z'] or any(coord in col_name.lower() for coord in [
                'coordinatex', 'coordinatey', 'coordinatez',
                'center_x', 'center_y', 'center_z',
                'center_x_nm', 'center_y_nm', 'center_z_nm',
                'point_x', 'point_y', 'point_z',
            ]):
                if '_nm' not in col_name and '_um' not in col_name:
                    return f"{col_name}_nm"
        
        # Already has units or doesn't need them
        return col_name
    
    def _flatten_results(self, data: Any, prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested dictionary/list structures to simple key-value pairs.
        Only includes numeric values and strings, skips complex structures.
        Special handling for individual_zone_results to create summary columns.
        """
        flattened = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}{key}" if prefix else key
                
                # Special handling for individual_zone_results
                if key == 'individual_zone_results' and isinstance(value, dict):
                    # Create summary statistics for individual zones instead of flattening all data
                    zone_count = len(value)
                    if zone_count > 0:
                        # Calculate summary stats across all zones
                        avg_cleft_widths = []
                        cleft_width_stds = []
                        measurement_counts = []
                        
                        for zone_name, zone_data in value.items():
                            if isinstance(zone_data, dict):
                                if 'average_cleft_width' in zone_data:
                                    avg_cleft_widths.append(zone_data['average_cleft_width'])
                                if 'cleft_width_std' in zone_data:
                                    cleft_width_stds.append(zone_data['cleft_width_std'])
                                if 'measurement_count' in zone_data:
                                    measurement_counts.append(zone_data['measurement_count'])
                        
                        # Skip adding summary columns for individual zones to avoid cluttering CSV
                        # Individual zone details are available in the JSON results
                        pass
                    
                    # Skip flattening the individual zone data to avoid long column names
                    continue
                
                # Handle other nested structures with shorter names
                if key == 'active_zone' and isinstance(value, dict):
                    # Flatten active zone data with shorter prefixes
                    nested = self._flatten_results(value, "")
                    flattened.update(nested)
                    continue
                elif key == 'cleft_width' and isinstance(value, dict):
                    # Flatten cleft width data with shorter prefixes, excluding total_measurements
                    nested = self._flatten_results(value, "")
                    # Remove total_measurements from the flattened results
                    nested.pop('total_measurements', None)
                    flattened.update(nested)
                    continue
                elif key == 'membrane_volumes' and isinstance(value, dict):
                    # Flatten membrane volumes data with shorter prefixes
                    nested = self._flatten_results(value, "")
                    flattened.update(nested)
                    continue
                
                if isinstance(value, (int, float, str, bool)):
                    # Add units to column names for numeric values
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        final_key = self._add_units_to_column_name(new_key)
                    else:
                        final_key = new_key
                    flattened[final_key] = value
                elif isinstance(value, (list, dict)):
                    # Recursively flatten nested structures
                    nested = self._flatten_results(value, f"{new_key}_")
                    flattened.update(nested)
        elif isinstance(data, list):
            # For lists, just count them or take first few values
            if data and isinstance(data[0], (int, float)):
                flattened[f"{prefix}count"] = len(data)
                if len(data) <= 5:  # Only include first 5 values
                    for i, val in enumerate(data):
                        flattened[f"{prefix}value_{i}"] = val
                else:
                    # Create a more CSV-friendly representation of the first 5 values
                    first_5_str = "[" + ", ".join([f"{val:.6f}" for val in data[:5]]) + "]"
                    flattened[f"{prefix}first_5_values"] = first_5_str
        
        return flattened
    
    def export_activezone_per_zone_csv(self) -> Optional[Path]:
        """Export active zone results as one row per tomogram + active zone."""
        rows: List[Dict[str, Any]] = []
        for results_key, analyses in self.results.items():
            if "activezone" not in analyses or "results" not in analyses["activezone"]:
                continue
            data = analyses["activezone"]
            results = data["results"]
            set_name = data.get("set_name", "") or ""
            alignment_dir = data.get("alignment_dir", "") or ""
            if "__" in results_key:
                tomogram_name, key_alignment = results_key.split("__", 1)
                if not alignment_dir:
                    alignment_dir = key_alignment
            else:
                tomogram_name = results_key

            from .activezone import build_activezone_per_zone_rows

            rows.extend(
                build_activezone_per_zone_rows(
                    tomogram_name=tomogram_name,
                    set_name=set_name,
                    alignment_dir=alignment_dir,
                    az_results=results.get("active_zone", {}),
                    cleft_results=results.get("cleft_width", {}),
                )
            )

        if not rows:
            return None

        csv_path = self.results_dir / "activezone" / "activezone_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Exported activezone results to {csv_path} ({len(rows)} zone rows)")
        return csv_path

    def export_aunps_per_zone_csv(self) -> Optional[Path]:
        """Export AuNP results as one row per tomogram + active zone."""
        rows: List[Dict[str, Any]] = []
        skipped = 0
        for results_key, analyses in self.results.items():
            if "aunps" not in analyses or "results" not in analyses["aunps"]:
                continue
            data = analyses["aunps"]
            results = data["results"]
            aunp_results = results.get("aunp_analysis", results)
            zone_results = aunp_results.get("individual_zone_results") or {}
            if not zone_results:
                skipped += 1
                continue

            set_name = data.get("set_name", "") or ""
            alignment_dir = data.get("alignment_dir", "") or ""
            if "__" in results_key:
                tomogram_name, key_alignment = results_key.split("__", 1)
                if not alignment_dir:
                    alignment_dir = key_alignment
            else:
                tomogram_name = results_key

            for zone_row in zone_results.values():
                if isinstance(zone_row, dict):
                    rows.append(dict(zone_row))

        if not rows:
            if skipped:
                print(
                    "No per-zone AuNP rows to export "
                    f"({skipped} tomogram(s) lack individual_zone_results; re-run AuNP analysis)."
                )
            return None

        csv_path = self.results_dir / "aunps" / "aunps_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.columns = [self._add_units_to_column_name(col) for col in df.columns]
        df.to_csv(csv_path, index=False)
        print(f"Exported aunps results to {csv_path} ({len(rows)} zone rows)")
        if skipped:
            print(f"  Skipped {skipped} tomogram(s) without per-zone AuNP data (re-run to include).")
        return csv_path

    def export_to_csv(self, output_file: Optional[str] = None):
        """Export results to separate CSV files for each analysis type."""
        # Group results by analysis type
        analysis_groups = {}
        
        for tomogram_name, analyses in self.results.items():
            for analysis_type, data in analyses.items():
                if analysis_type in ("activezone", "aunps"):
                    continue
                if 'results' in data:
                    if analysis_type not in analysis_groups:
                        analysis_groups[analysis_type] = []
                    
                    # Start with basic info
                    row = {
                        'tomogram_name': tomogram_name,
                        'set_name': data.get('set_name', ''),
                        'alignment_dir': data.get('alignment_dir', ''),
                        'timestamp': data.get('timestamp', ''),
                    }
                    if not row['alignment_dir'] and "__" in tomogram_name:
                        # Backward compatibility: parse legacy composite key.
                        row['alignment_dir'] = tomogram_name.split("__", 1)[1]
                    
                    # Flatten the results data
                    flattened_results = self._flatten_results(data['results'])
                    row.update(flattened_results)
                    
                    analysis_groups[analysis_type].append(row)
        
        # Export each analysis type to its own CSV file in step-specific subdirectories
        exported_files = []
        activezone_csv = self.export_activezone_per_zone_csv()
        if activezone_csv is not None:
            exported_files.append(activezone_csv)
        aunps_csv = self.export_aunps_per_zone_csv()
        if aunps_csv is not None:
            exported_files.append(aunps_csv)

        for analysis_type, rows in analysis_groups.items():
            if rows:
                try:
                    df = pd.DataFrame(rows)
                    
                    # Remove 'aunp_analysis_' prefix from column names for aunps results
                    if analysis_type == 'aunps':
                        df.columns = [col.replace('aunp_analysis_', '') if col.startswith('aunp_analysis_') else col for col in df.columns]
                    
                    # Add units to column names that don't already have them
                    df.columns = [self._add_units_to_column_name(col) for col in df.columns]
                    
                    csv_filename = f"{analysis_type}_results.csv"
                    # Save in step-specific subdirectory
                    step_dir = self.results_dir / analysis_type
                    step_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = step_dir / csv_filename
                    df.to_csv(csv_path, index=False)
                    exported_files.append(csv_path)
                    print(f"Exported {analysis_type} results to {csv_path}")
                except Exception as e:
                    print(f"Error exporting {analysis_type} results: {e}")
                    import traceback
                    traceback.print_exc()
        
        if exported_files:
            print(f"Exported {len(exported_files)} analysis files to {self.results_dir}")
        else:
            print("No results to export")
    
    def list_completed_analyses(self) -> Dict[str, List[str]]:
        """List all tomograms and their completed analyses."""
        completed = {}
        for tomogram_name, analyses in self.results.items():
            completed[tomogram_name] = list(analyses.keys())
        return completed
    
    def has_results(self, tomogram_name: str, analysis_type: str) -> bool:
        """Check if results exist for a specific tomogram and analysis."""
        return (tomogram_name in self.results and 
                analysis_type in self.results[tomogram_name])