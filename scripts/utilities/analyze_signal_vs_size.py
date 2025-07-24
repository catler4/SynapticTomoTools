#!/usr/bin/env python3
"""
Analyze the relationship between vesicle size (radius) and average signal.

This script uses results/all_vesicle_info.csv and the columns 'vesicle_radius_nm' (for size) and 'average_signal' (for signal).
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Load the vesicle info CSV
csv_path = 'results/all_vesicle_info.csv'
df = pd.read_csv(csv_path)

# Filter out rows with missing values
mask = df['average_signal'].notnull() & df['vesicle_radius_nm'].notnull()
df = df[mask]

if len(df) == 0:
    print("No vesicles with both radius and average signal found.")
    exit(1)

print(f"Analyzing {len(df)} vesicles with both radius and average signal.")
print(f"Radius range: {df['vesicle_radius_nm'].min():.2f} to {df['vesicle_radius_nm'].max():.2f}")
print(f"Signal range: {df['average_signal'].min():.4f} to {df['average_signal'].max():.4f}")

# Normalize average_signal per tomogram
if 'tomogram_name' not in df.columns:
    raise ValueError("Column 'tomogram_name' not found in CSV. Cannot normalize per tomogram.")

def normalize_group(group):
    min_val = group['average_signal'].min()
    max_val = group['average_signal'].max()
    if max_val > min_val:
        return (group['average_signal'] - min_val) / (max_val - min_val)
    else:
        return group['average_signal'] * 0  # all same value, set to 0

# Apply normalization per tomogram

df['normalized_signal'] = df.groupby('tomogram_name', group_keys=False).apply(normalize_group)

print(f"Normalized signal range: {df['normalized_signal'].min():.4f} to {df['normalized_signal'].max():.4f}")

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(df['vesicle_radius_nm'], df['normalized_signal'], alpha=0.5, edgecolor='k')
plt.xlabel('Vesicle Radius (nm)')
plt.ylabel('Normalized Average Signal (per tomogram)')
plt.title('Normalized Vesicle Signal vs. Vesicle Size')

# Fit and plot regression line
if len(df) > 1:
    m, b = np.polyfit(df['vesicle_radius_nm'], df['normalized_signal'], 1)
    y_fit = m * df['vesicle_radius_nm'] + b
    # Calculate R^2
    ss_res = np.sum((df['normalized_signal'] - y_fit) ** 2)
    ss_tot = np.sum((df['normalized_signal'] - df['normalized_signal'].mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    plt.plot(df['vesicle_radius_nm'], y_fit, color='red', label=f'Linear fit ($R^2$={r_squared:.3f})')
    plt.legend()

plt.tight_layout()
plt.savefig('vesicle_signal_vs_size_normalized.png', dpi=300)
print("Scatter plot saved as 'vesicle_signal_vs_size_normalized.png'")
plt.show()

# Calculate correlations
pearson_r, pearson_p = pearsonr(df['vesicle_radius_nm'], df['normalized_signal'])
spearman_r, spearman_p = spearmanr(df['vesicle_radius_nm'], df['normalized_signal'])

# Ensure p-values are floats for comparison
# Handle case where result is a tuple of tuples (e.g., ((r, p), ...))
def extract_float(val):
    if isinstance(val, tuple):
        return float(val[0])
    return float(val)

pearson_p = extract_float(pearson_p)
spearman_p = extract_float(spearman_p)

print(f"\nPearson correlation (normalized): r={pearson_r:.3f}, p={pearson_p:.3g}")
print(f"Spearman correlation (normalized): r={spearman_r:.3f}, p={spearman_p:.3g}")
if len(df) > 1:
    print(f"Linear fit R^2: {r_squared:.3f}")

if pearson_p < 0.05:
    print("Pearson correlation is statistically significant (p < 0.05)")
else:
    print("Pearson correlation is NOT statistically significant (p >= 0.05)")

if spearman_p < 0.05:
    print("Spearman correlation is statistically significant (p < 0.05)")
else:
    print("Spearman correlation is NOT statistically significant (p >= 0.05)") 