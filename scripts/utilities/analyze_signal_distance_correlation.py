#!/usr/bin/env python3
"""
Analyze the relationship between vesicle signal values and distance to active zone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def analyze_signal_distance_correlation():
    """Analyze the correlation between vesicle signal values and distance to active zone."""
    
    # Load the vesicle distances CSV
    df = pd.read_csv('results/all_vesicle_distances.csv')
    
    # Filter out rows with missing signal values
    df_with_signals = df.dropna(subset=['scaled_signal', 'distance_to_active_zone_nm'])
    
    if len(df_with_signals) == 0:
        print("No signal values found in the CSV file.")
        return
    
    # Extract data
    scaled_signals = df_with_signals['scaled_signal'].values
    distances = df_with_signals['distance_to_active_zone_nm'].values
    
    print(f"Analyzing {len(scaled_signals)} vesicles with both signal and distance data")
    print(f"Signal range: {scaled_signals.min():.4f} to {scaled_signals.max():.4f}")
    print(f"Distance range: {distances.min():.2f} to {distances.max():.2f} nm")
    
    # Calculate correlation coefficients
    print("\n=== Correlation Analysis ===")
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(scaled_signals, distances)
    print(f"Pearson correlation: r={pearson_r:.4f}, p-value={pearson_p:.4f}")
    print(f"  → {'Significant' if pearson_p < 0.05 else 'Not significant'} (α=0.05)")
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(scaled_signals, distances)
    print(f"Spearman correlation: r={spearman_r:.4f}, p-value={spearman_p:.4f}")
    print(f"  → {'Significant' if spearman_p < 0.05 else 'Not significant'} (α=0.05)")
    
    # Kendall correlation
    kendall_tau, kendall_p = stats.kendalltau(scaled_signals, distances)
    print(f"Kendall correlation: τ={kendall_tau:.4f}, p-value={kendall_p:.4f}")
    print(f"  → {'Significant' if kendall_p < 0.05 else 'Not significant'} (α=0.05)")
    
    # Linear regression
    print("\n=== Linear Regression Analysis ===")
    slope, intercept, r_value, p_value, std_err = stats.linregress(distances, scaled_signals)
    print(f"Linear regression: y = {slope:.6f}x + {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"  → {'Significant relationship' if p_value < 0.05 else 'No significant relationship'} (α=0.05)")
    
    # Analyze by distance groups
    print("\n=== Analysis by Distance Groups ===")
    
    # Group vesicles by distance ranges
    distance_ranges = [
        (0, 10, "0-10 nm"),
        (10, 50, "10-50 nm"),
        (50, 100, "50-100 nm"),
        (100, 200, "100-200 nm"),
        (200, 400, "200-400 nm")
    ]
    
    for min_dist, max_dist, label in distance_ranges:
        mask = (distances >= min_dist) & (distances < max_dist)
        group_signals = scaled_signals[mask]
        
        if len(group_signals) > 0:
            print(f"{label} ({len(group_signals)} vesicles):")
            print(f"  Mean signal: {group_signals.mean():.4f}")
            print(f"  Std signal: {group_signals.std():.4f}")
            print(f"  Signal range: {group_signals.min():.4f} to {group_signals.max():.4f}")
        else:
            print(f"{label}: No vesicles in this range")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    axes[0, 0].scatter(distances, scaled_signals, alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Distance to Active Zone (nm)')
    axes[0, 0].set_ylabel('Scaled Signal Value')
    axes[0, 0].set_title('Vesicle Signal vs Distance to Active Zone')
    
    # Add regression line
    x_range = np.linspace(distances.min(), distances.max(), 100)
    y_pred = slope * x_range + intercept
    axes[0, 0].plot(x_range, y_pred, 'r-', linewidth=2, 
                     label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')
    axes[0, 0].legend()
    
    # Box plot by distance groups
    distance_groups = []
    signal_groups = []
    group_labels = []
    
    for min_dist, max_dist, label in distance_ranges:
        mask = (distances >= min_dist) & (distances < max_dist)
        group_signals = scaled_signals[mask]
        
        if len(group_signals) > 0:
            distance_groups.append(group_signals)
            signal_groups.append(group_signals)
            group_labels.append(f"{label}\n(n={len(group_signals)})")
    
    if distance_groups:
        axes[0, 1].boxplot(distance_groups, labels=group_labels)
        axes[0, 1].set_ylabel('Scaled Signal Value')
        axes[0, 1].set_title('Signal Distribution by Distance Groups')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Violin plot
    if distance_groups:
        axes[1, 0].violinplot(distance_groups, positions=range(len(distance_groups)))
        axes[1, 0].set_xticks(range(len(distance_groups)))
        axes[1, 0].set_xticklabels(group_labels, rotation=45)
        axes[1, 0].set_ylabel('Scaled Signal Value')
        axes[1, 0].set_title('Signal Distribution by Distance Groups (Violin)')
    
    # Correlation heatmap
    correlation_matrix = np.corrcoef([scaled_signals, distances])
    im = axes[1, 1].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Signal', 'Distance'])
    axes[1, 1].set_yticklabels(['Signal', 'Distance'])
    axes[1, 1].set_title('Correlation Matrix')
    
    # Add correlation values to heatmap
    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig('vesicle_signal_distance_correlation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'vesicle_signal_distance_correlation.png'")
    
    # Summary conclusion
    print("\n=== Summary ===")
    if abs(pearson_r) > 0.3 and pearson_p < 0.05:
        direction = "positive" if pearson_r > 0 else "negative"
        print(f"There is a {direction} correlation between vesicle signal and distance to active zone")
    elif pearson_p < 0.05:
        print("There is a weak but significant correlation between vesicle signal and distance to active zone")
    else:
        print("There is no significant correlation between vesicle signal and distance to active zone")
    
    print(f"Correlation strength: {abs(pearson_r):.3f} ({'strong' if abs(pearson_r) > 0.7 else 'moderate' if abs(pearson_r) > 0.3 else 'weak'})")

if __name__ == "__main__":
    analyze_signal_distance_correlation() 