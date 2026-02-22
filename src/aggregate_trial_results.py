"""
Aggregate Trial Results Script
This script reads multiple trial CSV files from a folder and aggregates them
to compute average metrics and per-statement statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import glob
import re

def load_trial_results(folder_path):
    """Load all trial result CSV files from a folder."""
    # Find all files matching the pattern
    pattern = f"{folder_path}/conformal_results_trial*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No trial result files found in {folder_path}")
        return None
    
    print(f"Found {len(files)} trial result files")
    
    # Load all files
    dfs = []
    for file in sorted(files):
        # Extract trial number from filename
        match = re.search(r'trial(\d+)_', Path(file).name)
        if match:
            trial_num = int(match.group(1))
            df = pd.read_csv(file)
            df['trial'] = trial_num
            dfs.append(df)
            print(f"  Loaded {Path(file).name} ({len(df)} rows)")
    
    if not dfs:
        print("No valid trial files loaded")
        return None
    
    return dfs

def compute_trial_metrics(df):
    """Compute metrics for a single trial dataframe."""
    # Coverage
    coverage = df['is_covered'].mean()
    
    # Set size
    avg_set_size = df['set_size'].mean()
    
    # Singleton rate
    singleton_rate = (df['set_size'] == 1).mean()
    
    # Accuracy (if predicted_verdict matches true_verdict)
    accuracy = (df['predicted_verdict'] == df['true_verdict']).mean()
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'singleton_rate': singleton_rate,
        'accuracy': accuracy
    }

def aggregate_results(dfs, output_folder):
    """Aggregate results from multiple trial dataframes."""
    
    # Compute per-trial metrics
    print("\n" + "="*80)
    print("COMPUTING PER-TRIAL METRICS")
    print("="*80)
    
    trial_metrics = []
    for df in dfs:
        trial_num = df['trial'].iloc[0]
        metrics = compute_trial_metrics(df)
        metrics['trial'] = trial_num
        trial_metrics.append(metrics)
        print(f"Trial {trial_num}: Coverage={metrics['coverage']:.4f}, "
              f"Set Size={metrics['avg_set_size']:.4f}, "
              f"Accuracy={metrics['accuracy']:.4f}")
    
    # Aggregate trial-level metrics
    print("\n" + "="*80)
    print("AGGREGATED RESULTS ACROSS ALL TRIALS")
    print("="*80)
    
    avg_coverage = np.mean([m['coverage'] for m in trial_metrics])
    std_coverage = np.std([m['coverage'] for m in trial_metrics])
    avg_set_size = np.mean([m['avg_set_size'] for m in trial_metrics])
    std_set_size = np.std([m['avg_set_size'] for m in trial_metrics])
    avg_singleton = np.mean([m['singleton_rate'] for m in trial_metrics])
    std_singleton = np.std([m['singleton_rate'] for m in trial_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in trial_metrics])
    std_accuracy = np.std([m['accuracy'] for m in trial_metrics])
    
    print(f"\nPoint Prediction Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Marginal Coverage: {avg_coverage:.4f} ± {std_coverage:.4f}")
    print(f"Average Set Size: {avg_set_size:.4f} ± {std_set_size:.4f}")
    print(f"Singleton Rate: {avg_singleton:.4f} ± {std_singleton:.4f}")
    
    # Combine all trial results
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Aggregate per-statement results
    print("\n" + "="*80)
    print("AGGREGATING PER-STATEMENT RESULTS")
    print("="*80)
    
    statement_aggregated = combined_df.groupby('statement').agg({
        'set_size': ['mean', 'std', 'count'],
        'is_covered': 'mean',
        'true_verdict': 'first',
        'predicted_verdict': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common prediction
    }).reset_index()
    
    statement_aggregated.columns = ['statement', 'avg_set_size', 'std_set_size', 
                                     'num_trials', 'coverage_rate', 'true_verdict', 
                                     'most_common_prediction']
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_trials = len(dfs)
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Save per-statement aggregated results
    agg_file = f"{output_folder}/conformal_results_aggregated_{num_trials}trials_{timestamp}.csv"
    statement_aggregated.to_csv(agg_file, index=False)
    print(f"\nPer-statement aggregated results saved to: {agg_file}")
    
    # Save all individual trial results combined
    all_trials_file = f"{output_folder}/conformal_results_all_trials_{num_trials}trials_{timestamp}.csv"
    combined_df.to_csv(all_trials_file, index=False)
    print(f"All trial results saved to: {all_trials_file}")
    
    # Save trial-level metrics summary
    trial_metrics_df = pd.DataFrame(trial_metrics)
    metrics_file = f"{output_folder}/trial_metrics_summary_{num_trials}trials_{timestamp}.csv"
    trial_metrics_df.to_csv(metrics_file, index=False)
    print(f"Trial metrics summary saved to: {metrics_file}")
    
    # Print summary statistics
    print(f"\n" + "="*80)
    print("PER-STATEMENT STATISTICS")
    print("="*80)
    print(f"Total unique statements evaluated: {len(statement_aggregated)}")
    print(f"Average prediction set size across all statements: {statement_aggregated['avg_set_size'].mean():.4f}")
    print(f"Average coverage rate across all statements: {statement_aggregated['coverage_rate'].mean():.4f}")
    
    # Print statements with highest/lowest coverage
    print("\nStatements with LOWEST coverage rate:")
    lowest_coverage = statement_aggregated.nsmallest(5, 'coverage_rate')[['statement', 'coverage_rate', 'avg_set_size', 'true_verdict']]
    for idx, row in lowest_coverage.iterrows():
        print(f"  Coverage: {row['coverage_rate']:.2f}, Set Size: {row['avg_set_size']:.2f}, "
              f"Verdict: {row['true_verdict']}")
        print(f"  Statement: {row['statement'][:100]}...")
    
    print("\nStatements with HIGHEST coverage rate:")
    highest_coverage = statement_aggregated.nlargest(5, 'coverage_rate')[['statement', 'coverage_rate', 'avg_set_size', 'true_verdict']]
    for idx, row in highest_coverage.iterrows():
        print(f"  Coverage: {row['coverage_rate']:.2f}, Set Size: {row['avg_set_size']:.2f}, "
              f"Verdict: {row['true_verdict']}")
        print(f"  Statement: {row['statement'][:100]}...")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate trial results from multiple CSV files."
    )
    parser.add_argument(
        "--input-folder",
        default="results/zero-shot/jobs",
        help="Folder containing trial result CSV files.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder to save aggregated results (defaults to same as input folder).",
    )
    
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder if args.output_folder else input_folder
    
    print("="*80)
    print("Trial Results Aggregation Script")
    print("="*80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Load trial results
    dfs = load_trial_results(input_folder)
    
    if dfs is None or len(dfs) == 0:
        print("No trial results to aggregate. Exiting.")
        return
    
    # Aggregate results
    aggregate_results(dfs, output_folder)
    
    print("\n" + "="*80)
    print("Aggregation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
