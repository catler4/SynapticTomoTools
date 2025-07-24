#!/usr/bin/env python3
"""
Analyze the distribution of vesicle signal values to determine if they follow a Gaussian distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson
import seaborn as sns

def analyze_signal_distribution():
    """Analyze the distribution of vesicle signal values."""
    
    # Load the vesicle distances CSV
    df = pd.read_csv('results/all_vesicle_distances.csv')
    
    # Filter out rows with missing signal values
    df_with_signals = df.dropna(subset=['scaled_signal'])
    
    if len(df_with_signals) == 0:
        print("No signal values found in the CSV file.")
        return
    
    # Extract scaled signal values
    scaled_signals = df_with_signals['scaled_signal'].values
    
    print(f"Analyzing {len(scaled_signals)} vesicle signal values")
    print(f"Signal range: {scaled_signals.min():.4f} to {scaled_signals.max():.4f}")
    print(f"Mean: {scaled_signals.mean():.4f}")
    print(f"Standard deviation: {scaled_signals.std():.4f}")
    print(f"Median: {np.median(scaled_signals):.4f}")
    
    # Test for normality using multiple methods
    print("\n=== Normality Tests ===")
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(scaled_signals)
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    print(f"  → {'Normal' if shapiro_p > 0.05 else 'Not normal'} (α=0.05)")
    
    # D'Agostino K^2 test
    dagostino_stat, dagostino_p = normaltest(scaled_signals)
    print(f"D'Agostino K^2 test: statistic={dagostino_stat:.4f}, p-value={dagostino_p:.4f}")
    print(f"  → {'Normal' if dagostino_p > 0.05 else 'Not normal'} (α=0.05)")
    
    # Anderson-Darling test
    anderson_result = anderson(scaled_signals)
    print(f"Anderson-Darling test: statistic={anderson_result.statistic:.4f}")
    print(f"  → {'Normal' if anderson_result.statistic < anderson_result.critical_values[2] else 'Not normal'} (α=0.05)")
    
    # Q-Q plot analysis
    print("\n=== Q-Q Plot Analysis ===")
    qq_stat, qq_p = stats.probplot(scaled_signals, dist="norm")
    print(f"Q-Q plot correlation: {np.corrcoef(qq_stat[0], qq_stat[1])[0,1]:.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram with normal curve overlay
    axes[0, 0].hist(scaled_signals, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    x = np.linspace(scaled_signals.min(), scaled_signals.max(), 100)
    normal_curve = stats.norm.pdf(x, scaled_signals.mean(), scaled_signals.std())
    axes[0, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal distribution')
    axes[0, 0].set_title('Histogram with Normal Curve Overlay')
    axes[0, 0].set_xlabel('Scaled Signal Values')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # Q-Q plot
    stats.probplot(scaled_signals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Box plot
    axes[1, 0].boxplot(scaled_signals)
    axes[1, 0].set_title('Box Plot')
    axes[1, 0].set_ylabel('Scaled Signal Values')
    
    # Violin plot
    sns.violinplot(y=scaled_signals, ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot')
    axes[1, 1].set_ylabel('Scaled Signal Values')
    
    plt.tight_layout()
    plt.savefig('vesicle_signal_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'vesicle_signal_distribution_analysis.png'")
    
    # Additional statistics
    print("\n=== Additional Statistics ===")
    print(f"Skewness: {stats.skew(scaled_signals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(scaled_signals):.4f}")
    
    # Check if skewness and kurtosis are within normal range
    skew_normal = abs(stats.skew(scaled_signals)) < 1.0
    kurtosis_normal = abs(stats.kurtosis(scaled_signals)) < 2.0
    print(f"Skewness within normal range (±1.0): {'Yes' if skew_normal else 'No'}")
    print(f"Kurtosis within normal range (±2.0): {'Yes' if kurtosis_normal else 'No'}")
    
    # Summary conclusion
    print("\n=== Summary ===")
    normal_tests_passed = sum([
        shapiro_p > 0.05,
        dagostino_p > 0.05,
        anderson_result.statistic < anderson_result.critical_values[2],
        skew_normal,
        kurtosis_normal
    ])
    
    if normal_tests_passed >= 4:
        print("The vesicle signal distribution appears to be GAUSSIAN/NORMAL")
    elif normal_tests_passed >= 2:
        print("The vesicle signal distribution is MODERATELY GAUSSIAN")
    else:
        print("The vesicle signal distribution is NOT GAUSSIAN")
    
    print(f"({normal_tests_passed}/5 normality criteria met)")

if __name__ == "__main__":
    analyze_signal_distribution() 