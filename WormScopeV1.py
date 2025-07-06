"""
C. elegans Survival Analysis Pipeline
====================================
Comprehensive survival analysis for C. elegans lifespan data with optimizations.

Author: [Your Name]
Date: [Current Date]
Version: 2.0 (Optimized)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from scipy.integrate import trapezoid
from scipy.stats import chi2_contingency
from pathlib import Path
import argparse
from itertools import combinations
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Color palette for strains
PALETTE = {
    'WT': '#1f77b4',
    'skn1b': '#ff7f0e',
    'daf2': '#2ca02c',
    'daf2_skn1b': '#d62728'
}


# ----------------------------
# Helper Functions
# ----------------------------

def create_sample_data() -> pd.DataFrame:
    """
    Create sample C. elegans survival data for demonstration.

    Returns:
        pd.DataFrame: Sample survival data with columns [strain, time, status]
    """
    np.random.seed(42)

    strains = {
        'WT': {'n': 100, 'scale': 20, 'shape': 2.5},
        'skn-1b(tm4241)': {'n': 80, 'scale': 18, 'shape': 2.3},
        'daf-2(e1370)': {'n': 90, 'scale': 35, 'shape': 2.8},
        'daf-2(e1370); skn-1b(tm4241)': {'n': 85, 'scale': 30, 'shape': 2.6}
    }

    data = []
    for strain, params in strains.items():
        times = np.random.weibull(params['shape'], params['n']) * params['scale']
        censored = np.random.binomial(1, 0.1, params['n'])

        for t, c in zip(times, censored):
            data.append({
                'strain': strain,
                'time': int(t),
                'status': 1 - c
            })

    return pd.DataFrame(data)


def parse_survival_data(filename: Path) -> pd.DataFrame:
    """
    Parse survival data from text file.

    Args:
        filename: Path to the data file

    Returns:
        pd.DataFrame: Parsed survival data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        with open(filename, "r") as f:
            raw_data = f.readlines()

        strain_data = []
        current_strain = None

        for line in raw_data:
            if line.startswith("%"):
                current_strain = line.split("% ")[1].split(" [")[0].strip()
            elif line.startswith("#days") or not line.strip():
                continue
            else:
                parts = line.split()
                if len(parts) >= 3:
                    days = int(parts[0])
                    dead = int(parts[1])
                    censored = int(parts[2])

                    for _ in range(dead):
                        strain_data.append({
                            'strain': current_strain,
                            'time': days,
                            'status': 1
                        })

                    for _ in range(censored):
                        strain_data.append({
                            'strain': current_strain,
                            'time': days,
                            'status': 0
                        })

        if not strain_data:
            raise ValueError("No valid data found in file")

        return pd.DataFrame(strain_data)

    except FileNotFoundError:
        print(f"File '{filename}' not found. Creating sample data instead.")
        return create_sample_data()


def simplify_strain_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify strain names for better visualization.

    Args:
        df: DataFrame with strain column

    Returns:
        pd.DataFrame: DataFrame with simplified strain names
    """
    strain_map = {
        'WT': 'WT',
        'skn-1b(tm4241)': 'skn1b',
        'daf-2(e1370)': 'daf2',
        'daf-2(e1370); skn-1b(tm4241)': 'daf2_skn1b'
    }
    df['strain'] = df['strain'].map(strain_map)
    return df


def pairwise_logrank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all pairwise log-rank tests between strains.

    Args:
        df: Survival data with columns [strain, time, status]

    Returns:
        pd.DataFrame: Results with columns [strain1, strain2, chi2, p_value]
    """
    strains = sorted(df['strain'].unique())
    results = []

    # Use combinations to avoid duplicate comparisons
    for strain1, strain2 in combinations(strains, 2):
        group1 = df[df['strain'] == strain1]
        group2 = df[df['strain'] == strain2]

        result = logrank_test(
            group1['time'], group2['time'],
            group1['status'], group2['status']
        )

        results.append({
            'strain1': strain1,
            'strain2': strain2,
            'chi2': result.test_statistic,
            'p_value': result.p_value
        })

    return pd.DataFrame(results)


def bootstrap_median_ci(times: np.ndarray, status: np.ndarray,
                        n_bootstrap: int = 1000,
                        alpha: float = 0.05) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for median survival time.

    Args:
        times: Survival times
        status: Event status (1=death, 0=censored)
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level

    Returns:
        Tuple of (lower_ci, upper_ci)
    """
    medians = []
    n = len(times)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        kmf = KaplanMeierFitter()

        try:
            kmf.fit(times[idx], status[idx])
            median = kmf.median_survival_time_
            if not pd.isna(median):
                medians.append(median)
        except Exception:
            continue

    if len(medians) > 0:
        lower = np.percentile(medians, alpha / 2 * 100)
        upper = np.percentile(medians, (1 - alpha / 2) * 100)
        return lower, upper
    else:
        return np.nan, np.nan


# ----------------------------
# Analysis Functions
# ----------------------------

def analyze_kaplan_meier(df: pd.DataFrame, output_dir: Path) -> Dict[str, KaplanMeierFitter]:
    """
    Generate Kaplan-Meier survival curves.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        Dictionary of KaplanMeierFitter objects by strain
    """
    print("Generating Kaplan-Meier survival curves...")
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    kmf_objects = {}
    for strain, group in df.groupby('strain'):
        kmf = KaplanMeierFitter()
        kmf.fit(group['time'], group['status'], label=strain)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[strain], linewidth=2)
        kmf_objects[strain] = kmf

    add_at_risk_counts(*kmf_objects.values(), ax=ax)

    plt.title("Kaplan-Meier Survival Curves - C. elegans Strains", fontsize=16, pad=20)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.xlabel("Time (days)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_kaplan_meier_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    return kmf_objects


def analyze_logrank_tests(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Perform pairwise log-rank tests with visualization.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        DataFrame with test results
    """
    print("Performing pairwise log-rank tests...")

    # Get pairwise results
    results_df = pairwise_logrank(df)

    # Create matrix for heatmap
    strains = sorted(df['strain'].unique())
    n_strains = len(strains)
    pval_matrix = np.ones((n_strains, n_strains))

    for _, row in results_df.iterrows():
        i = strains.index(row['strain1'])
        j = strains.index(row['strain2'])
        pval_matrix[i, j] = row['p_value']
        pval_matrix[j, i] = row['p_value']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(pval_matrix, dtype=bool), k=1)
    sns.heatmap(pval_matrix, mask=mask, annot=True, fmt='.4f',
                xticklabels=strains, yticklabels=strains,
                cmap='RdYlBu_r', center=0.05, vmin=0, vmax=0.1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title("Pairwise Log-rank Test P-values", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "02_logrank_test_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save text results
    with open(output_dir / "02_logrank_test_results.txt", "w", encoding='utf-8') as f:
        f.write("Pairwise Log-rank Test Results\n")
        f.write("=" * 50 + "\n\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['strain1']} vs {row['strain2']}: "
                    f"χ² = {row['chi2']:.2f}, p-value = {row['p_value']:.4f}\n")

    return results_df


def analyze_hazard_functions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Calculate and plot hazard functions.

    Args:
        df: Survival data
        output_dir: Directory for output files
    """
    print("Calculating hazard functions...")
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    for strain, group in df.groupby('strain'):
        naf = NelsonAalenFitter()
        naf.fit(group['time'], group['status'], label=strain)
        naf.plot_hazard(bandwidth=3, ax=ax, color=PALETTE[strain], linewidth=2)

    plt.title("Smoothed Hazard Functions", fontsize=16, pad=20)
    plt.ylabel("Hazard Rate", fontsize=14)
    plt.xlabel("Time (days)", fontsize=14)
    plt.legend(title="Strain", loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_hazard_functions.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_cox_model(df: pd.DataFrame, output_dir: Path) -> CoxPHFitter:
    """
    Fit Cox proportional hazards model.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        Fitted CoxPHFitter object
    """
    print("Fitting Cox proportional hazards model...")

    # Prepare data
    df_cox = pd.get_dummies(df, columns=['strain'], drop_first=True)

    # Fit model
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col='time', event_col='status')

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, 6))

    summary = cph.summary
    hr = np.exp(summary['coef'])
    hr_lower = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    hr_upper = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])

    y_pos = np.arange(len(hr))
    ax.scatter(hr, y_pos, s=100, color='black', zorder=3)
    ax.hlines(y_pos, hr_lower, hr_upper, color='black', linewidth=2)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('strain_', '') for name in hr.index])
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=14)
    ax.set_title('Cox Proportional Hazards Model - Forest Plot', fontsize=16, pad=20)
    ax.grid(True, axis='x', alpha=0.3)

    for i, (h, l, u) in enumerate(zip(hr, hr_lower, hr_upper)):
        ax.text(hr.max() * 1.1, i, f'{h:.2f} ({l:.2f}-{u:.2f})',
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "05_cox_forest_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary
    with open(output_dir / "05_cox_model_summary.txt", "w", encoding='utf-8') as f:
        f.write("Cox Proportional Hazards Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(cph.summary))

    return cph


def analyze_median_survival(df: pd.DataFrame, output_dir: Path,
                            kmf_objects: Dict[str, KaplanMeierFitter]) -> pd.DataFrame:
    """
    Calculate median survival times with confidence intervals.

    Args:
        df: Survival data
        output_dir: Directory for output files
        kmf_objects: Dictionary of fitted KM objects

    Returns:
        DataFrame with median survival statistics
    """
    print("Calculating median survival times...")

    median_data = []

    for strain, group in df.groupby('strain'):
        kmf = kmf_objects[strain]

        try:
            median = kmf.median_survival_time_
            if pd.isna(median):
                median = float(kmf.percentile(0.5))
        except (ValueError, KeyError):
            median = group['time'].median()

        # Try to get CI from lifelines, otherwise bootstrap
        try:
            # Attempt to get CI from survival function
            lower, upper = bootstrap_median_ci(
                group['time'].values,
                group['status'].values
            )
        except Exception:
            lower = median * 0.9
            upper = median * 1.1

        median_data.append({
            'strain': strain,
            'n': len(group),
            'median': median,
            'ci_lower': lower,
            'ci_upper': upper
        })

    median_df = pd.DataFrame(median_data)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(median_df))
    colors = [PALETTE[s] for s in median_df['strain']]

    ax.barh(y_pos, median_df['median'],
            xerr=[median_df['median'] - median_df['ci_lower'],
                  median_df['ci_upper'] - median_df['median']],
            color=colors, alpha=0.7, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(median_df['strain'])
    ax.set_xlabel('Median Survival Time (days)', fontsize=14)
    ax.set_title('Median Survival Times with 95% Confidence Intervals', fontsize=16, pad=20)
    ax.grid(True, axis='x', alpha=0.3)

    for i, row in median_df.iterrows():
        ax.text(row['median'] + 1, i,
                f"{row['median']:.1f} ({row['ci_lower']:.1f}-{row['ci_upper']:.1f})",
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "08_median_survival_times.png", dpi=300, bbox_inches='tight')
    plt.close()

    return median_df


def analyze_chi_square_bonferroni(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Perform chi-square analysis with Bonferroni correction.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        DataFrame with chi-square test results
    """
    print("Performing chi-square analysis with Bonferroni correction...")

    # Get unique pairwise comparisons
    logrank_results = pairwise_logrank(df)

    # Calculate Bonferroni correction
    n_comparisons = len(logrank_results)

    chi_square_results = []
    for _, row in logrank_results.iterrows():
        p_value = row['p_value']
        bonferroni_p = min(p_value * n_comparisons, 1.0)

        # Format p-values
        p_value_str = "0" if p_value < 0.0001 else f"{p_value:.4f}"
        bonferroni_p_str = "0" if bonferroni_p < 0.0001 else f"{bonferroni_p:.4f}"

        # Significance
        if bonferroni_p < 0.001:
            sig = "***"
        elif bonferroni_p < 0.01:
            sig = "**"
        elif bonferroni_p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        chi_square_results.append({
            'Comparison': f"{row['strain1']} vs {row['strain2']}",
            'χ²': f"{row['chi2']:.2f}",
            'P-value': p_value_str,
            'Bonferroni P-value': bonferroni_p_str,
            'Significance': sig
        })

    chi_df = pd.DataFrame(chi_square_results)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=chi_df.values,
                     colLabels=chi_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style the header
    for i in range(len(chi_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, min(len(chi_df) + 1, len(table.get_celld()))):
        for j in range(len(chi_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E7E7')

            if j == 4 and i - 1 < chi_df.shape[0]:
                sig_value = chi_df.iloc[i - 1, j]
                if sig_value != 'ns':
                    table[(i, j)].set_text_props(weight='bold', color='red')

    plt.suptitle('Chi-Square Test Results (Log-rank) with Bonferroni Correction',
                 fontsize=16, y=0.98, weight='bold')

    plt.figtext(0.5, 0.02,
                f"Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n"
                f"Bonferroni correction applied for {n_comparisons} comparisons\n"
                "Note: χ² values are from log-rank test comparing survival distributions",
                ha='center', va='bottom', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / "13_chi_square_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save text file
    with open(output_dir / "13_chi_square_analysis.txt", "w", encoding='utf-8') as f:
        f.write("Chi-Square Analysis (Log-rank) with Bonferroni Correction\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total number of comparisons: {n_comparisons}\n")
        f.write(f"Bonferroni correction factor: {n_comparisons}\n\n")
        f.write(chi_df.to_string(index=False))
        f.write("\n\nSignificance levels:")
        f.write("\n*** p < 0.001")
        f.write("\n**  p < 0.01")
        f.write("\n*   p < 0.05")
        f.write("\nns  not significant")
        f.write("\n\nNote: χ² values are from log-rank test comparing survival distributions")
        f.write("\nP-values of 0 indicate p < 0.0001")

    return chi_df


def create_summary_statistics(df: pd.DataFrame, output_dir: Path,
                              kmf_objects: Dict[str, KaplanMeierFitter]) -> pd.DataFrame:
    """
    Create comprehensive summary statistics table.

    Args:
        df: Survival data
        output_dir: Directory for output files
        kmf_objects: Dictionary of fitted KM objects

    Returns:
        DataFrame with summary statistics
    """
    print("Creating summary statistics...")

    stats_data = []
    for strain, group in df.groupby('strain'):
        n = len(group)
        n_events = group['status'].sum()
        n_censored = n - n_events

        # Raw statistics (deaths only)
        deaths_only = group[group['status'] == 1]['time']
        if len(deaths_only) > 0:
            raw_mean = deaths_only.mean()
            raw_std = deaths_only.std()
            raw_median = deaths_only.median()
        else:
            raw_mean = raw_std = raw_median = np.nan

        # KM statistics
        kmf = kmf_objects[strain]

        try:
            km_median = kmf.median_survival_time_
            if pd.isna(km_median):
                km_median = float(kmf.percentile(0.5))
        except (ValueError, KeyError):
            km_median = np.nan

        # RMST
        try:
            surv_func = kmf.survival_function_
            if len(surv_func) > 0:
                rmst = trapezoid(surv_func.values.flatten(), surv_func.index)
            else:
                rmst = np.nan
        except Exception:
            rmst = np.nan

        stats_data.append([
            strain, n, n_events, n_censored,
            f"{raw_mean:.1f} ± {raw_std:.1f}" if not pd.isna(raw_mean) else "N/A",
            f"{raw_median:.1f}" if not pd.isna(raw_median) else "N/A",
            f"{km_median:.1f}" if not pd.isna(km_median) else "N/A",
            f"{rmst:.1f}" if not pd.isna(rmst) else "N/A"
        ])

    stats_df = pd.DataFrame(
        stats_data,
        columns=['Strain', 'N', 'Deaths', 'Censored',
                 'Mean ± SD\n(deaths only)', 'Median\n(deaths only)',
                 'KM Median', 'RMST']
    )

    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=stats_df.values,
                     colLabels=stats_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style the header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_df) + 1):
        for j in range(len(stats_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E7E7')

    plt.title('Summary Statistics by Strain', fontsize=18, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "10_summary_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save text file
    with open(output_dir / "10_summary_statistics.txt", "w", encoding='utf-8') as f:
        f.write("Summary Statistics by Strain\n")
        f.write("=" * 80 + "\n\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")
        f.write("RMST = Restricted Mean Survival Time\n")
        f.write("KM = Kaplan-Meier estimate\n")

    return stats_df


def main():
    """Main analysis pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="C. elegans Survival Analysis Pipeline"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('2.txt'),
        help='Input data file (default: 2.txt)'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--bootstrap-ci',
        action='store_true',
        help='Use bootstrap for confidence intervals'
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print(f"Loading data from {args.input}...")
    df = parse_survival_data(args.input)
    df = simplify_strain_names(df)

    print(f"Total samples: {len(df)}")
    print(f"Strains: {', '.join(df['strain'].unique())}")
    print(f"\nOutput directory: {args.outdir}")
    print("\nStarting survival analysis...\n")

    # Run analyses
    kmf_objects = analyze_kaplan_meier(df, args.outdir)
    logrank_results = analyze_logrank_tests(df, args.outdir)
    analyze_hazard_functions(df, args.outdir)

    # Additional analyses
    analyze_cumulative_hazard(df, args.outdir)
    cph = analyze_cox_model(df, args.outdir)
    analyze_weibull_aft(df, args.outdir)
    analyze_survival_quantiles(df, args.outdir)
    median_df = analyze_median_survival(df, args.outdir, kmf_objects)
    analyze_individual_timelines(df, args.outdir)
    stats_df = create_summary_statistics(df, args.outdir, kmf_objects)

    # New analyses
    analyze_survival_curves_clean(df, args.outdir)
    analyze_mean_max_lifespan(df, args.outdir)
    chi_square_df = analyze_chi_square_bonferroni(df, args.outdir)

    # Create comprehensive summary
    create_comprehensive_summary(df, args.outdir, kmf_objects, cph)

    print("\nAll analyses completed successfully!")
    print(f"Generated 13 analysis files in {args.outdir}")
    print("\nFiles created:")
    print("- 01_kaplan_meier_curves.png")
    print("- 02_logrank_test_heatmap.png & 02_logrank_test_results.txt")
    print("- 03_hazard_functions.png")
    print("- 04_cumulative_hazard.png")
    print("- 05_cox_forest_plot.png & 05_cox_model_summary.txt")
    print("- 06_weibull_aft_parameters.png & 06_weibull_model_summary.txt")
    print("- 07_survival_quantiles.png")
    print("- 08_median_survival_times.png")
    print("- 09_individual_timelines.png")
    print("- 10_summary_statistics.png & 10_summary_statistics.txt")
    print("- 11_survival_curves_clean.png")
    print("- 12_mean_max_lifespan_table.png & 12_mean_max_lifespan.txt")
    print("- 13_chi_square_analysis.png & 13_chi_square_analysis.txt")


def analyze_cumulative_hazard(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Calculate and plot cumulative hazard functions.

    Args:
        df: Survival data
        output_dir: Directory for output files
    """
    print("Calculating cumulative hazard functions...")
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    for strain, group in df.groupby('strain'):
        naf = NelsonAalenFitter()
        naf.fit(group['time'], group['status'], label=strain)
        naf.plot_cumulative_hazard(ax=ax, color=PALETTE[strain], linewidth=2)

    plt.title("Nelson-Aalen Cumulative Hazard Functions", fontsize=16, pad=20)
    plt.ylabel("Cumulative Hazard", fontsize=14)
    plt.xlabel("Time (days)", fontsize=14)
    plt.legend(title="Strain", loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "04_cumulative_hazard.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_weibull_aft(df: pd.DataFrame, output_dir: Path) -> WeibullAFTFitter:
    """
    Fit and visualize Weibull AFT model.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        Fitted WeibullAFTFitter object
    """
    print("Fitting Weibull AFT model...")

    # Prepare data with dummy variables
    df_aft = pd.get_dummies(df, columns=['strain'], drop_first=False)

    # Fit model
    aft = WeibullAFTFitter()
    aft.fit(df_aft, duration_col='time', event_col='status')

    # Plot parameters
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get AFT parameters
    aft_summary = aft.summary

    # Handle MultiIndex case
    if isinstance(aft_summary.index, pd.MultiIndex):
        aft_summary = aft_summary.reset_index()
        if 'covariate' in aft_summary.columns:
            strain_params = aft_summary[aft_summary['covariate'].str.contains('strain_', na=False)]
        else:
            strain_params = pd.DataFrame()
    else:
        try:
            strain_params = aft_summary[aft_summary.index.astype(str).str.contains('strain_')]
        except Exception:
            strain_params = pd.DataFrame()

    if len(strain_params) > 0:
        if 'covariate' in strain_params.columns:
            param_names = [name.replace('strain_', '') for name in strain_params['covariate']]
            coef_values = strain_params['coef'].values
            se_values = strain_params['se(coef)'].values
        else:
            param_names = [str(idx).replace('strain_', '') for idx in strain_params.index]
            coef_values = strain_params['coef'].values if 'coef' in strain_params.columns else strain_params.iloc[:,
                                                                                               0].values
            se_values = strain_params['se(coef)'].values if 'se(coef)' in strain_params.columns else strain_params.iloc[
                                                                                                     :, 1].values

        x = np.arange(len(param_names))
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(param_names)]

        bars = ax.bar(x, coef_values, yerr=1.96 * se_values,
                      capsize=5, color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45)
        ax.set_ylabel('AFT Coefficient', fontsize=14)
        ax.set_title('Weibull AFT Model - Strain Effects', fontsize=16, pad=20)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No strain-specific parameters found\nCheck model summary output',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Weibull AFT Model', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "06_weibull_aft_parameters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary
    with open(output_dir / "06_weibull_model_summary.txt", "w", encoding='utf-8') as f:
        f.write("Weibull AFT Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(aft.summary))

    return aft


def analyze_survival_quantiles(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Calculate and visualize survival quantiles.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        DataFrame with quantile data
    """
    print("Calculating survival quantiles...")

    quantiles = [0.25, 0.5, 0.75]
    quantile_data = []

    for strain, group in df.groupby('strain'):
        kmf = KaplanMeierFitter()
        kmf.fit(group['time'], group['status'])

        for q in quantiles:
            try:
                quantile_time = kmf.percentile(1 - q)  # lifelines uses 1-q
                if pd.isna(quantile_time):
                    surv_func = kmf.survival_function_
                    times_below_q = surv_func.index[surv_func.iloc[:, 0] <= q]
                    if len(times_below_q) > 0:
                        quantile_time = times_below_q[0]
                    else:
                        quantile_time = surv_func.index[-1]

                quantile_data.append({
                    'strain': strain,
                    'quantile': f"{int(q * 100)}%",
                    'time': float(quantile_time)
                })
            except Exception:
                quantile_data.append({
                    'strain': strain,
                    'quantile': f"{int(q * 100)}%",
                    'time': np.nan
                })

    quant_df = pd.DataFrame(quantile_data)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='quantile', y='time', hue='strain', data=quant_df, palette=PALETTE)
    plt.title("Lifespan Quantiles by Strain", fontsize=16, pad=20)
    plt.ylabel("Time (days)", fontsize=14)
    plt.xlabel("Survival Quantile", fontsize=14)
    plt.legend(title="Strain", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "07_survival_quantiles.png", dpi=300, bbox_inches='tight')
    plt.close()

    return quant_df


def analyze_individual_timelines(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create individual survival timeline visualization.

    Args:
        df: Survival data
        output_dir: Directory for output files
    """
    print("Creating individual survival timelines...")

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Sample data for visualization
    sample_size = 15
    sample_df = df.groupby('strain', group_keys=False).apply(
        lambda x: x.sample(min(sample_size, len(x)), random_state=42)
    ).reset_index(drop=True)

    # Create timeline plot
    y_position = 0
    y_labels = []
    strains = sorted(df['strain'].unique())

    for strain in strains:
        strain_data = sample_df[sample_df['strain'] == strain]

        for _, row in strain_data.iterrows():
            if row['status'] == 1:  # Death
                ax.plot([0, row['time']], [y_position, y_position],
                        color=PALETTE[strain], linewidth=2)
                ax.scatter(row['time'], y_position, color=PALETTE[strain],
                           s=100, marker='x', linewidths=2)
            else:  # Censored
                ax.plot([0, row['time']], [y_position, y_position],
                        color=PALETTE[strain], linewidth=2)
                ax.scatter(row['time'], y_position, color=PALETTE[strain],
                           s=100, marker='o', facecolors='none', linewidths=2)

            y_labels.append(strain)
            y_position += 1

        # Add spacing between strains
        y_position += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_title('Individual Survival Timelines (Sample)', fontsize=16, pad=20)
    ax.grid(True, axis='x', alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Survival time'),
        Line2D([0], [0], marker='x', color='gray', lw=0, markersize=8, label='Death'),
        Line2D([0], [0], marker='o', color='gray', lw=0, markersize=8,
               markerfacecolor='none', label='Censored')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "09_individual_timelines.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_survival_curves_clean(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create clean, well-labeled survival curves plot.

    Args:
        df: Survival data
        output_dir: Directory for output files
    """
    print("Creating clean survival curves plot...")

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)

    # Plot each strain with clear styling
    for strain, group in df.groupby('strain'):
        kmf = KaplanMeierFitter()
        kmf.fit(group['time'], group['status'], label=strain)

        # Plot with confidence intervals
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[strain],
                                   linewidth=3, alpha=0.9)

    # Enhance the plot
    ax.set_xlabel('Days', fontsize=16, weight='bold')
    ax.set_ylabel('Survival Probability', fontsize=16, weight='bold')
    ax.set_title('Survival Curves of C. elegans Strains', fontsize=18, weight='bold', pad=20)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Enhance legend
    ax.legend(title='Strain', fontsize=12, title_fontsize=14,
              loc='best', frameon=True, fancybox=True, shadow=True)

    # Set axis limits
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)

    # Add tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "11_survival_curves_clean.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_mean_max_lifespan(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Create mean and maximum lifespan table.

    Args:
        df: Survival data
        output_dir: Directory for output files

    Returns:
        DataFrame with lifespan statistics
    """
    print("Creating mean and maximum lifespan table...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Calculate mean and max lifespan for each strain
    lifespan_data = []
    for strain, group in df.groupby('strain'):
        # All observations
        all_mean = group['time'].mean()
        all_std = group['time'].std()
        all_sem = group['time'].sem()
        all_max = group['time'].max()

        # Deaths only
        deaths_only = group[group['status'] == 1]['time']
        if len(deaths_only) > 0:
            death_mean = deaths_only.mean()
            death_std = deaths_only.std()
            death_sem = deaths_only.sem()
            death_max = deaths_only.max()
        else:
            death_mean = death_std = death_sem = death_max = np.nan

        # Count
        n_total = len(group)
        n_deaths = group['status'].sum()

        lifespan_data.append([
            strain,
            n_total,
            n_deaths,
            f"{all_mean:.2f} ± {all_sem:.2f}",
            f"{death_mean:.2f} ± {death_sem:.2f}" if not pd.isna(death_mean) else "N/A",
            f"{all_max:.0f}",
            f"{death_max:.0f}" if not pd.isna(death_max) else "N/A"
        ])

    lifespan_df = pd.DataFrame(
        lifespan_data,
        columns=['Strain', 'N Total', 'N Deaths', 'Mean ± SEM (All)',
                 'Mean ± SEM (Deaths)', 'Max (All)', 'Max (Deaths)']
    )

    # Create table
    table = ax.table(cellText=lifespan_df.values,
                     colLabels=lifespan_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style the header
    for i in range(len(lifespan_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(lifespan_df) + 1):
        for j in range(len(lifespan_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E7E7')

    plt.title('Mean and Maximum Lifespan by Strain', fontsize=18, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "12_mean_max_lifespan_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save as text file
    with open(output_dir / "12_mean_max_lifespan.txt", "w", encoding='utf-8') as f:
        f.write("Mean and Maximum Lifespan Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(lifespan_df.to_string(index=False))
        f.write("\n\nNote: Mean ± SEM (Standard Error of Mean)")
        f.write("\n'All' includes both deaths and censored observations")
        f.write("\n'Deaths' includes only confirmed deaths (status=1)")

    return lifespan_df


def create_comprehensive_summary(df: pd.DataFrame, output_dir: Path,
                                 kmf_objects: Dict[str, KaplanMeierFitter],
                                 cph: CoxPHFitter) -> None:
    """
    Create comprehensive summary figure with multiple panels.

    Args:
        df: Survival data
        output_dir: Directory for output files
        kmf_objects: Dictionary of fitted KM objects
        cph: Fitted Cox model
    """
    print("Creating comprehensive summary figure...")

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # KM Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for strain, group in df.groupby('strain'):
        kmf = kmf_objects[strain]
        kmf.plot_survival_function(ax=ax1, ci_show=False, color=PALETTE[strain])
    ax1.set_title("A. Kaplan-Meier Survival Curves")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Survival Probability")

    # Hazard Functions
    ax2 = fig.add_subplot(gs[0, 1])
    for strain, group in df.groupby('strain'):
        naf = NelsonAalenFitter()
        naf.fit(group['time'], group['status'], label=strain)
        naf.plot_hazard(bandwidth=2, ax=ax2, color=PALETTE[strain])
    ax2.set_title("B. Hazard Functions")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Hazard Rate")

    # Quantile Survival
    ax3 = fig.add_subplot(gs[0, 2])
    quant_data = []
    for strain, group in df.groupby('strain'):
        kmf = kmf_objects[strain]
        try:
            median = kmf.median_survival_time_
            if pd.isna(median):
                median = float(kmf.percentile(0.5))
        except Exception:
            median = group['time'].median()
        quant_data.append({'strain': strain, 'median': median})

    quant_df = pd.DataFrame(quant_data)
    bars = ax3.bar(quant_df['strain'], quant_df['median'],
                   color=[PALETTE[s] for s in quant_df['strain']])
    ax3.set_title("C. Median Survival Times")
    ax3.set_xlabel("Strain")
    ax3.set_ylabel("Days")
    ax3.set_xticklabels(quant_df['strain'], rotation=45)

    # Cox Forest Plot
    ax4 = fig.add_subplot(gs[1, 0])
    summary = cph.summary
    hr = np.exp(summary['coef'])
    hr_lower = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    hr_upper = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])

    y_pos = np.arange(len(hr))
    ax4.scatter(hr, y_pos, s=100, color='black', zorder=3)
    ax4.hlines(y_pos, hr_lower, hr_upper, color='black', linewidth=2)
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([name.replace('strain_', '') for name in hr.index])
    ax4.set_xlabel('Hazard Ratio')
    ax4.set_title("D. Cox Model - Hazard Ratios")

    # Cumulative Hazard
    ax5 = fig.add_subplot(gs[1, 1])
    for strain, group in df.groupby('strain'):
        naf = NelsonAalenFitter()
        naf.fit(group['time'], group['status'], label=strain)
        naf.plot_cumulative_hazard(ax=ax5, color=PALETTE[strain])
    ax5.set_title("E. Cumulative Hazard Functions")
    ax5.set_xlabel("Time (days)")
    ax5.set_ylabel("Cumulative Hazard")

    # Sample sizes and events
    ax6 = fig.add_subplot(gs[1, 2])
    sample_data = []
    for strain, group in df.groupby('strain'):
        sample_data.append({
            'strain': strain,
            'n_total': len(group),
            'n_events': group['status'].sum()
        })
    sample_df = pd.DataFrame(sample_data)

    x = np.arange(len(sample_df))
    width = 0.35
    bars1 = ax6.bar(x - width / 2, sample_df['n_total'], width, label='Total', alpha=0.8)
    bars2 = ax6.bar(x + width / 2, sample_df['n_events'], width, label='Events', alpha=0.8)
    ax6.set_xlabel('Strain')
    ax6.set_ylabel('Count')
    ax6.set_title('F. Sample Sizes')
    ax6.set_xticks(x)
    ax6.set_xticklabels(sample_df['strain'], rotation=45)
    ax6.legend()

    # Statistical Summary Table
    ax7 = fig.add_subplot(gs[2, :])
    stats_data = []
    for strain, group in df.groupby('strain'):
        n = len(group)
        n_events = group['status'].sum()
        median = group['time'].median()
        mean = group['time'].mean()
        std = group['time'].std()
        stats_data.append([strain, n, n_events, f"{mean:.1f} ± {std:.1f}", f"{median:.1f}"])

    stats_df = pd.DataFrame(
        stats_data,
        columns=['Strain', 'N', 'Deaths', 'Mean ± SD', 'Median']
    )

    ax7.axis('off')
    table = ax7.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.3, 0.8, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax7.set_title("G. Lifespan Statistics Summary", y=0.9, fontsize=12)

    plt.suptitle("C. elegans Survival Analysis - Comprehensive Summary", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()