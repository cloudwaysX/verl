#!/usr/bin/env python3
"""
Script to analyze correlations between scores at different passes and lengths.
Tests whether:
1. edit_score with weight 0.4 at pass k can predict performance at the next higher pass
2. regular mean_score at pass k can predict performance at the next higher pass
3. edit_score with weight 0.4 at length m can predict performance at the next higher length
4. regular mean_score at length m can predict performance at the next higher length

With detailed analysis of performance across different score ranges.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from scipy.stats import pearsonr, spearmanr
from tabulate import tabulate  # You might need to install this: pip install tabulate

def extract_info_from_path(filepath):
    """
    Extract model configuration, length and pass information from filepath.
    Expected format: .../deepscaler-1.5b-2k_forceans/pass2_analysis.csv
    """
    path_parts = filepath.split('/')
    
    # Look for the folder with the model configuration
    for i in range(len(path_parts)-1, 0, -1):
        if 'deepscaler' in path_parts[i] and '-' in path_parts[i]:
            model_folder = path_parts[i]
            # Extract length from the model configuration
            length_match = re.search(r'-(\d+)k', model_folder)
            if length_match:
                length = int(length_match.group(1)) * 1000
                break
    else:
        # If we didn't find a matching folder
        return None, None
        
    # Extract pass number from filename
    filename = os.path.basename(filepath)
    pass_match = re.search(r'pass(\d+)', filename)
    pass_num = int(pass_match.group(1)) if pass_match else None
    
    return pass_num, length

def load_analysis_files(base_dir, pattern="**/pass*_analysis.csv"):
    """Load all analysis files matching pattern from base directory and subdirectories."""
    files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    pass_dataframes = {}
    
    print(f"Found {len(files)} analysis files")
    
    for file in files:
        pass_num, length = extract_info_from_path(file)
        
        # Skip if we couldn't extract valid pass number or length
        if pass_num is None or length is None:
            print(f"Warning: Could not extract pass or length from {file}, skipping")
            continue
            
        df = pd.read_csv(file)
        
        # Store dataframe by length and pass number
        if length not in pass_dataframes:
            pass_dataframes[length] = {}
        pass_dataframes[length][pass_num] = df
    
    return pass_dataframes

def analyze_next_level_correlations(data_dict, weight=0.4):
    """
    Analyze correlations between different predictors at one level and performance at the next level
    for both passes and lengths.
    
    Compare:
    1. edit_score_weight as predictor
    2. mean_score as predictor
    """
    # Define the sequence of passes and lengths
    pass_sequence = [1, 2, 4, 6, 8]
    length_sequence = [2000, 4000]
    
    # Results storage
    pass_results = []
    length_results = []
    
    # For detailed analysis by score range
    pass_detailed_data = []
    length_detailed_data = []
    
    # Analyze next-level pass correlations
    for i in range(len(pass_sequence) - 1):
        base_pass = pass_sequence[i]
        next_pass = pass_sequence[i + 1]
        
        for length in length_sequence:
            if length not in data_dict or base_pass not in data_dict[length] or next_pass not in data_dict[length]:
                continue
                
            base_df = data_dict[length][base_pass]
            next_df = data_dict[length][next_pass]
            
            # Check if required columns exist
            if f"mean_edit_score_{weight}" not in base_df.columns or "mean_score" not in base_df.columns:
                continue
                
            # Take the minimum number of rows to ensure alignment
            num_rows = len(base_df)
            if num_rows < 2:
                continue
                
            # Get values for both predictors
            edit_scores = base_df[f"mean_edit_score_{weight}"]
            mean_scores = base_df["mean_score"] 
            # Target values to predict
            next_scores = next_df["mean_score"]
            
            # Calculate correlations for edit scores
            edit_pearson, edit_p_value = pearsonr(edit_scores, next_scores)
            
            # Calculate correlations for mean scores
            mean_pearson, mean_p_value = pearsonr(mean_scores, next_scores)
                
            pass_results.append({
                "length": length,
                "base_pass": base_pass,
                "next_pass": next_pass,
                "weight": weight,
                "edit_score_pearson": edit_pearson,
                "edit_score_p_value": edit_p_value,
                "mean_score_pearson": mean_pearson,
                "mean_score_p_value": mean_p_value,
                "n_samples": num_rows
            })
            
            # Store detailed data for score range analysis
            for j in range(num_rows):
                pass_detailed_data.append({
                    "length": length,
                    "base_pass": base_pass,
                    "next_pass": next_pass,
                    "edit_score": edit_scores.iloc[j],
                    "mean_score": mean_scores.iloc[j],
                    "next_score": next_scores.iloc[j],
                    "edit_error": abs(edit_scores.iloc[j] - next_scores.iloc[j]),
                    "mean_error": abs(mean_scores.iloc[j] - next_scores.iloc[j]),
                    "better_predictor": "edit_score" if abs(edit_scores.iloc[j] - next_scores.iloc[j]) < 
                                                       abs(mean_scores.iloc[j] - next_scores.iloc[j]) else "mean_score"
                })
    
    # Analyze next-level length correlations
    for i in range(len(length_sequence) - 1):
        base_length = length_sequence[i]
        next_length = length_sequence[i + 1]
        
        for pass_num in pass_sequence:
            if (base_length not in data_dict or next_length not in data_dict or 
                pass_num not in data_dict[base_length] or pass_num not in data_dict[next_length]):
                continue
                
            base_df = data_dict[base_length][pass_num]
            next_df = data_dict[next_length][pass_num]
            
            # Check if required columns exist
            if f"mean_edit_score_{weight}" not in base_df.columns or "mean_score" not in base_df.columns:
                continue
                
            # Take the minimum number of rows to ensure alignment
            num_rows = len(base_df)
            if num_rows < 2:
                continue
                
            # Get values for both predictors
            edit_scores = base_df[f"mean_edit_score_{weight}"]
            mean_scores = base_df["mean_score"]
            # Target values to predict
            next_scores = next_df["mean_score"]
            
            # Calculate correlations for edit scores
            edit_pearson, edit_p_value = pearsonr(edit_scores, next_scores)
            
            # Calculate correlations for mean scores
            mean_pearson, mean_p_value = pearsonr(mean_scores, next_scores)
                
            length_results.append({
                "base_length": base_length,
                "next_length": next_length,
                "pass": pass_num,
                "weight": weight,
                "edit_score_pearson": edit_pearson,
                "edit_score_p_value": edit_p_value,
                "mean_score_pearson": mean_pearson,
                "mean_score_p_value": mean_p_value,
                "n_samples": num_rows
            })
            
            # Store detailed data for score range analysis
            for j in range(num_rows):
                length_detailed_data.append({
                    "base_length": base_length,
                    "next_length": next_length,
                    "pass": pass_num,
                    "edit_score": edit_scores.iloc[j],
                    "mean_score": mean_scores.iloc[j],
                    "next_score": next_scores.iloc[j],
                    "edit_error": abs(edit_scores.iloc[j] - next_scores.iloc[j]),
                    "mean_error": abs(mean_scores.iloc[j] - next_scores.iloc[j]),
                    "better_predictor": "edit_score" if abs(edit_scores.iloc[j] - next_scores.iloc[j]) < 
                                                       abs(mean_scores.iloc[j] - next_scores.iloc[j]) else "mean_score"
                })
    
    pass_results_df = pd.DataFrame(pass_results) if pass_results else None
    length_results_df = pd.DataFrame(length_results) if length_results else None
    pass_detailed_df = pd.DataFrame(pass_detailed_data) if pass_detailed_data else None
    length_detailed_df = pd.DataFrame(length_detailed_data) if length_detailed_data else None
    
    return pass_results_df, length_results_df, pass_detailed_df, length_detailed_df

def display_pass_correlations_table(df, weight):
    """Display pass correlation results as a table in the terminal."""
    if df is None or df.empty:
        print("No pass correlation data available.")
        return
    
    # Create a view with only the columns we want to show
    view_df = df[["base_pass", "next_pass", "length", "edit_score_pearson", "mean_score_pearson", "n_samples"]].copy()
    
    # Format the length column to show as "2k" instead of 2000
    view_df["length"] = view_df["length"].apply(lambda x: f"{int(x/1000)}k")
    
    # Format the correlation columns to show rounded values
    view_df["edit_score_pearson"] = view_df["edit_score_pearson"].apply(lambda x: f"{x:.3f}")
    view_df["mean_score_pearson"] = view_df["mean_score_pearson"].apply(lambda x: f"{x:.3f}")
    
    # Create table for all pass correlations
    print(f"\nCorrelations between predictors at base pass and scores at next pass:")
    print(tabulate(view_df, headers=["Base Pass", "Next Pass", "Length", f"Edit Score {weight}", "Mean Score", "Samples"], 
                  tablefmt="grid"))
    
    # Add column showing which predictor is better
    df['better_predictor'] = df.apply(
        lambda row: 'Edit Score' if abs(row['edit_score_pearson']) > abs(row['mean_score_pearson']) else 'Mean Score', 
        axis=1
    )
    
    # Count how many times each predictor is better
    better_counts = df['better_predictor'].value_counts()
    print("\nBetter predictor counts for pass prediction:")
    for predictor, count in better_counts.items():
        print(f"  {predictor}: {count} times")
    
    # Group by base_pass and next_pass and show average correlations
    grouped = df.groupby(["base_pass", "next_pass"]).agg({
        "edit_score_pearson": "mean",
        "mean_score_pearson": "mean",
        "n_samples": "sum"
    }).reset_index()
    
    # Format the correlation columns
    grouped["edit_score_pearson"] = grouped["edit_score_pearson"].apply(lambda x: f"{x:.3f}")
    grouped["mean_score_pearson"] = grouped["mean_score_pearson"].apply(lambda x: f"{x:.3f}")
    
    print(f"\nAverage correlations by pass pair:")
    print(tabulate(grouped, headers=["Base Pass", "Next Pass", f"Avg Edit Score {weight}", "Avg Mean Score", "Total Samples"], 
                  tablefmt="grid"))

def display_length_correlations_table(df, weight):
    """Display length correlation results as a table in the terminal."""
    if df is None or df.empty:
        print("No length correlation data available.")
        return
    
    # Create a view with only the columns we want to show
    view_df = df[["base_length", "next_length", "pass", "edit_score_pearson", "mean_score_pearson", "n_samples"]].copy()
    
    # Format the length columns to show as "2k" instead of 2000
    view_df["base_length"] = view_df["base_length"].apply(lambda x: f"{int(x/1000)}k")
    view_df["next_length"] = view_df["next_length"].apply(lambda x: f"{int(x/1000)}k")
    
    # Format the correlation columns to show rounded values
    view_df["edit_score_pearson"] = view_df["edit_score_pearson"].apply(lambda x: f"{x:.3f}")
    view_df["mean_score_pearson"] = view_df["mean_score_pearson"].apply(lambda x: f"{x:.3f}")
    
    # Create table for all length correlations
    print(f"\nCorrelations between predictors at base length and scores at next length:")
    print(tabulate(view_df, headers=["Base Length", "Next Length", "Pass", f"Edit Score {weight}", "Mean Score", "Samples"], 
                  tablefmt="grid"))
    
    # Add column showing which predictor is better
    df['better_predictor'] = df.apply(
        lambda row: 'Edit Score' if abs(row['edit_score_pearson']) > abs(row['mean_score_pearson']) else 'Mean Score', 
        axis=1
    )
    
    # Count how many times each predictor is better
    better_counts = df['better_predictor'].value_counts()
    print("\nBetter predictor counts for length prediction:")
    for predictor, count in better_counts.items():
        print(f"  {predictor}: {count} times")
    
    # Group by base_length and next_length and show average correlations
    grouped = df.groupby(["base_length", "next_length"]).agg({
        "edit_score_pearson": "mean",
        "mean_score_pearson": "mean",
        "n_samples": "sum"
    }).reset_index()
    
    # Format the length columns
    grouped["base_length"] = grouped["base_length"].apply(lambda x: f"{int(x/1000)}k")
    grouped["next_length"] = grouped["next_length"].apply(lambda x: f"{int(x/1000)}k")
    
    # Format the correlation columns
    grouped["edit_score_pearson"] = grouped["edit_score_pearson"].apply(lambda x: f"{x:.3f}")
    grouped["mean_score_pearson"] = grouped["mean_score_pearson"].apply(lambda x: f"{x:.3f}")
    
    print(f"\nAverage correlations by length pair:")
    print(tabulate(grouped, headers=["Base Length", "Next Length", f"Avg Edit Score {weight}", "Avg Mean Score", "Total Samples"], 
                  tablefmt="grid"))

def analyze_by_score_range(detailed_df, predictor_type="pass"):
    """
    Analyze prediction accuracy by score range.
    
    Parameters:
    detailed_df - DataFrame with detailed prediction data
    predictor_type - 'pass' or 'length' to determine which analysis to run
    """
    if detailed_df is None or detailed_df.empty:
        print(f"No detailed {predictor_type} data available for score range analysis.")
        return
    
    # Define score ranges for analysis
    score_ranges = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    
    # Results for each score range
    results = []
    
    for start, end in score_ranges:
        # Filter data for this score range - we'll analyze both by edit_score and mean_score ranges
        edit_range_mask = (detailed_df['edit_score'] >= start) & (detailed_df['edit_score'] < end)
        mean_range_mask = (detailed_df['mean_score'] >= start) & (detailed_df['mean_score'] < end)
        
        # For edit score range
        if edit_range_mask.sum() > 0:
            edit_range_data = detailed_df[edit_range_mask]
            
            # Calculate metrics
            edit_better_count = (edit_range_data['better_predictor'] == 'edit_score').sum()
            total_count = len(edit_range_data)
            edit_win_rate = edit_better_count / total_count if total_count > 0 else 0
            
            avg_edit_error = edit_range_data['edit_error'].mean()
            avg_mean_error = edit_range_data['mean_error'].mean()
            
            results.append({
                'score_range': f"{start:.1f}-{end:.1f}",
                'score_type': 'edit_score',
                'sample_count': total_count,
                'edit_better_count': edit_better_count,
                'edit_win_rate': edit_win_rate,
                'avg_edit_error': avg_edit_error,
                'avg_mean_error': avg_mean_error,
                'error_diff': avg_mean_error - avg_edit_error  # Positive means edit is better
            })
        
        # For mean score range
        if mean_range_mask.sum() > 0:
            mean_range_data = detailed_df[mean_range_mask]
            
            # Calculate metrics
            edit_better_count = (mean_range_data['better_predictor'] == 'edit_score').sum()
            total_count = len(mean_range_data)
            edit_win_rate = edit_better_count / total_count if total_count > 0 else 0
            
            avg_edit_error = mean_range_data['edit_error'].mean()
            avg_mean_error = mean_range_data['mean_error'].mean()
            
            results.append({
                'score_range': f"{start:.1f}-{end:.1f}",
                'score_type': 'mean_score',
                'sample_count': total_count,
                'edit_better_count': edit_better_count,
                'edit_win_rate': edit_win_rate,
                'avg_edit_error': avg_edit_error, 
                'avg_mean_error': avg_mean_error,
                'error_diff': avg_mean_error - avg_edit_error  # Positive means edit is better
            })
    
    # Convert to DataFrame for easy display
    results_df = pd.DataFrame(results)
    
    # Display results
    print(f"\nPrediction accuracy by score range for {predictor_type} prediction:")
    
    # Format numeric columns
    results_df['edit_win_rate'] = results_df['edit_win_rate'].apply(lambda x: f"{x:.2%}")
    results_df['avg_edit_error'] = results_df['avg_edit_error'].apply(lambda x: f"{x:.3f}")
    results_df['avg_mean_error'] = results_df['avg_mean_error'].apply(lambda x: f"{x:.3f}")
    results_df['error_diff'] = results_df['error_diff'].apply(lambda x: f"{x:.3f}")
    
    # Split and display by score type
    for score_type in ['edit_score', 'mean_score']:
        type_df = results_df[results_df['score_type'] == score_type].copy()
        
        print(f"\nAnalysis by {score_type} range:")
        print(tabulate(type_df, headers=[
            "Score Range", "Score Type", "Samples", "Edit Better", "Edit Win Rate", 
            "Avg Edit Error", "Avg Mean Error", "Error Diff"
        ], tablefmt="grid"))
    
    return results_df

def create_text_bar_chart(data, title, max_bar_length=40):
    """Create a simple bar chart using text characters."""
    max_value = max(data.values())
    
    print(f"\n{title}")
    print("-" * 50)
    
    for label, value in data.items():
        bar_length = int((value / max_value) * max_bar_length)
        bar = "█" * bar_length
        print(f"{label:15} | {value:.3f} | {bar}")
    
    print("-" * 50)

def main():
    # Configuration - modify these parameters as needed
    base_dir = "/mnt/disk3/verl/eval/deepscaler_5k/"  # Replace with your actual directory
    output_dir = "correlation_results"
    weight = 0.4
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all analysis files
    data_dict = load_analysis_files(base_dir)
    if not data_dict:
        print("No analysis files found with valid length and pass information. Check the directory path.")
        return
    
    print(f"Found data for {len(data_dict)} different lengths:")
    for length, passes in data_dict.items():
        print(f"  Length {int(length)/1000}k: {len(passes)} pass configurations - {sorted(passes.keys())}")
    
    # Analyze correlations between one level and the next higher level
    print(f"\nAnalyzing correlations with different predictors (edit_score_{weight} vs mean_score)...")
    pass_results, length_results, pass_detailed, length_detailed = analyze_next_level_correlations(data_dict, weight=weight)
    
    # Save results to CSV
    if pass_results is not None:
        pass_results.to_csv(os.path.join(output_dir, f"next_pass_correlations_comparison.csv"), index=False)
        print(f"Pass correlation results saved to CSV ({len(pass_results)} pairs analyzed)")
        # Display correlation tables
        display_pass_correlations_table(pass_results, weight)
    else:
        print("No valid pass correlation pairs found")
    
    if length_results is not None:
        length_results.to_csv(os.path.join(output_dir, f"next_length_correlations_comparison.csv"), index=False)
        print(f"Length correlation results saved to CSV ({len(length_results)} pairs analyzed)")
        # Display correlation tables
        display_length_correlations_table(length_results, weight)
    else:
        print("No valid length correlation pairs found")
    
    # Detailed analysis by score range
    if pass_detailed is not None:
        pass_range_results = analyze_by_score_range(pass_detailed, "pass")
        pass_detailed.to_csv(os.path.join(output_dir, f"pass_detailed_analysis.csv"), index=False)
    
    if length_detailed is not None:
        length_range_results = analyze_by_score_range(length_detailed, "length")
        length_detailed.to_csv(os.path.join(output_dir, f"length_detailed_analysis.csv"), index=False)
    
    # Create text bar charts for visualization
    print("\nVisualizing prediction accuracy:")
    
    # For pass prediction
    if pass_detailed is not None:
        # Overall win rate by score range (for edit scores)
        win_rates = {}
        for start, end in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            mask = (pass_detailed['edit_score'] >= start) & (pass_detailed['edit_score'] < end)
            if mask.sum() > 0:
                win_rate = (pass_detailed.loc[mask, 'better_predictor'] == 'edit_score').mean()
                win_rates[f"{start:.1f}-{end:.1f}"] = win_rate
        
        create_text_bar_chart(win_rates, "Edit Score Win Rate by Edit Score Range (Pass Prediction)")
        
        # Average error by pass 
        avg_errors_by_pass = {}
        for base_pass in sorted(pass_detailed['base_pass'].unique()):
            mask = pass_detailed['base_pass'] == base_pass
            if mask.sum() > 0:
                edit_error = pass_detailed.loc[mask, 'edit_error'].mean()
                mean_error = pass_detailed.loc[mask, 'mean_error'].mean()
                avg_errors_by_pass[f"Pass {base_pass} Edit"] = edit_error
                avg_errors_by_pass[f"Pass {base_pass} Mean"] = mean_error
        
        create_text_bar_chart(avg_errors_by_pass, "Average Prediction Error by Base Pass (lower is better)")
    
    # For length prediction
    if length_detailed is not None:
        # Overall win rate by score range (for edit scores)
        win_rates = {}
        for start, end in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            mask = (length_detailed['edit_score'] >= start) & (length_detailed['edit_score'] < end)
            if mask.sum() > 0:
                win_rate = (length_detailed.loc[mask, 'better_predictor'] == 'edit_score').mean()
                win_rates[f"{start:.1f}-{end:.1f}"] = win_rate
        
        create_text_bar_chart(win_rates, "Edit Score Win Rate by Edit Score Range (Length Prediction)")
        
        # Average error by pass for length prediction 
        avg_errors_by_pass = {}
        for pass_num in sorted(length_detailed['pass'].unique()):
            mask = length_detailed['pass'] == pass_num
            if mask.sum() > 0:
                edit_error = length_detailed.loc[mask, 'edit_error'].mean()
                mean_error = length_detailed.loc[mask, 'mean_error'].mean()
                avg_errors_by_pass[f"Pass {pass_num} Edit"] = edit_error
                avg_errors_by_pass[f"Pass {pass_num} Mean"] = mean_error
        
        create_text_bar_chart(avg_errors_by_pass, "Average Prediction Error by Pass (Length Prediction)")
        
        # Error difference by score range (positive means edit score is better)
        error_diff_by_range = {}
        for start, end in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            mask = (length_detailed['edit_score'] >= start) & (length_detailed['edit_score'] < end)
            if mask.sum() > 0:
                mean_error = length_detailed.loc[mask, 'mean_error'].mean()
                edit_error = length_detailed.loc[mask, 'edit_error'].mean()
                error_diff = mean_error - edit_error  # Positive means edit score is better
                error_diff_by_range[f"{start:.1f}-{end:.1f}"] = error_diff
        
        create_text_bar_chart(error_diff_by_range, "Error Difference by Edit Score Range (Positive = Edit is Better)")
    
    # Print summary of findings
    print("\nSummary of findings:")
    
    if pass_results is not None and not pass_results.empty:
        # Overall averages for both predictors
        avg_edit_corr = pass_results["edit_score_pearson"].mean()
        avg_mean_corr = pass_results["mean_score_pearson"].mean()
        print(f"- Average correlation for pass prediction:")
        print(f"  * Edit Score {weight}: {avg_edit_corr:.3f}")
        print(f"  * Mean Score: {avg_mean_corr:.3f}")
        print(f"  * Better predictor: {'Edit Score' if avg_edit_corr > avg_mean_corr else 'Mean Score'}")
        
        # Best base pass for each predictor
        edit_avg_by_base = pass_results.groupby("base_pass")["edit_score_pearson"].mean()
        edit_best_base_pass = edit_avg_by_base.idxmax()
        
        mean_avg_by_base = pass_results.groupby("base_pass")["mean_score_pearson"].mean()
        mean_best_base_pass = mean_avg_by_base.idxmax()
        
        print(f"- Best base pass for prediction:")
        print(f"  * Edit Score {weight}: pass {edit_best_base_pass} (avg r={edit_avg_by_base[edit_best_base_pass]:.3f})")
        print(f"  * Mean Score: pass {mean_best_base_pass} (avg r={mean_avg_by_base[mean_best_base_pass]:.3f})")
        
        # Best pair for each predictor
        edit_best_pair = pass_results.loc[pass_results["edit_score_pearson"].abs().idxmax()]
        mean_best_pair = pass_results.loc[pass_results["mean_score_pearson"].abs().idxmax()]
        
        print(f"- Strongest pass correlations:")
        print(f"  * Edit Score {weight}: pass {int(edit_best_pair.base_pass)} → pass {int(edit_best_pair.next_pass)} "
              f"at length {int(edit_best_pair.length)/1000}k (r={edit_best_pair.edit_score_pearson:.3f})")
        print(f"  * Mean Score: pass {int(mean_best_pair.base_pass)} → pass {int(mean_best_pair.next_pass)} "
              f"at length {int(mean_best_pair.length)/1000}k (r={mean_best_pair.mean_score_pearson:.3f})")
    
    if length_results is not None and not length_results.empty:
        # Overall averages for both predictors
        avg_edit_corr = length_results["edit_score_pearson"].mean()
        avg_mean_corr = length_results["mean_score_pearson"].mean()
        print(f"\n- Average correlation for length prediction:")
        print(f"  * Edit Score {weight}: {avg_edit_corr:.3f}")
        print(f"  * Mean Score: {avg_mean_corr:.3f}")
        print(f"  * Better predictor: {'Edit Score' if avg_edit_corr > avg_mean_corr else 'Mean Score'}")
        
        # Best base length for each predictor
        edit_avg_by_base = length_results.groupby("base_length")["