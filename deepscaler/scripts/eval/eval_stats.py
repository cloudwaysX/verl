#!/usr/bin/env python3
"""
Script to analyze correlations between edit scores at different passes and lengths.
Tests whether:
1. edit_score with weight 0.4 at pass k can predict performance at the next higher pass
2. edit_score with weight 0.4 at length m can predict performance at the next higher length
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
    Analyze correlations between edit_score at one level and performance at the next level
    for both passes and lengths.
    """
    # Define the sequence of passes and lengths
    pass_sequence = [1, 2, 4, 6, 8]
    length_sequence = [2000, 4000, 8000]
    
    # Results storage
    pass_results = []
    length_results = []
    
    # Analyze next-level pass correlations
    for i in range(len(pass_sequence) - 1):
        base_pass = pass_sequence[i]
        next_pass = pass_sequence[i + 1]
        
        for length in length_sequence:
            if length not in data_dict or base_pass not in data_dict[length] or next_pass not in data_dict[length]:
                continue
                
            base_df = data_dict[length][base_pass]
            next_df = data_dict[length][next_pass]
            
            if f"mean_edit_score_{weight}" not in base_df.columns:
                continue
                
            # Take the minimum number of rows to ensure alignment
            num_rows = len(base_df)
            if num_rows < 2:
                continue
                
            base_scores = base_df[f"mean_edit_score_{weight}"]
            next_scores = next_df["mean_score"]
            
            # Calculate correlations
            pearson, p_value = pearsonr(base_scores, next_scores)
            
                
            pass_results.append({
                "length": length,
                "base_pass": base_pass,
                "next_pass": next_pass,
                "weight": weight,
                "pearson_r": pearson,
                "pearson_p": p_value,
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
            
            if f"mean_edit_score_{weight}" not in base_df.columns:
                continue
                
            # Take the minimum number of rows to ensure alignment
            num_rows = len(base_df)
            if num_rows < 2:
                continue
                
            base_scores = base_df[f"mean_edit_score_{weight}"]
            next_scores = next_df["mean_score"]
            
            # Calculate correlations
            pearson, p_value = pearsonr(base_scores, next_scores)
            
                
            length_results.append({
                "base_length": base_length,
                "next_length": next_length,
                "pass": pass_num,
                "weight": weight,
                "pearson_r": pearson,
                "pearson_p": p_value,
                "n_samples": num_rows
            })
    
    return pd.DataFrame(pass_results) if pass_results else None, pd.DataFrame(length_results) if length_results else None

def display_pass_correlations_table(df, weight):
    """Display pass correlation results as a table in the terminal."""
    if df is None or df.empty:
        print("No pass correlation data available.")
        return
    
    # Create a view with only the columns we want to show
    view_df = df[["base_pass", "next_pass", "length", "pearson_r", "pass_pearson_r", "n_samples"]].copy()
    
    # Format the length column to show as "2k" instead of 2000
    view_df["length"] = view_df["length"].apply(lambda x: f"{int(x/1000)}k")
    
    # Format the correlation columns to show rounded values
    view_df["pearson_r"] = view_df["pearson_r"].apply(lambda x: f"{x:.3f}")
    view_df["pass_pearson_r"] = view_df["pass_pearson_r"].apply(lambda x: f"{x:.3f}")
    
    # Create table for all pass correlations
    print(f"\nCorrelations between edit_score_{weight} at base pass and scores at next pass:")
    print(tabulate(view_df, headers="keys", tablefmt="grid"))
    
    # Group by base_pass and next_pass and show average correlations
    grouped = df.groupby(["base_pass", "next_pass"]).agg({
        "pearson_r": "mean",
        "pass_pearson_r": "mean",
        "n_samples": "sum"
    }).reset_index()
    
    # Format the correlation columns
    grouped["pearson_r"] = grouped["pearson_r"].apply(lambda x: f"{x:.3f}")
    grouped["pass_pearson_r"] = grouped["pass_pearson_r"].apply(lambda x: f"{x:.3f}")
    
    print(f"\nAverage correlations by pass pair:")
    print(tabulate(grouped, headers=["Base Pass", "Next Pass", "Avg Pearson r", "Avg Pass Pearson r", "Total Samples"], 
                  tablefmt="grid"))

def display_length_correlations_table(df, weight):
    """Display length correlation results as a table in the terminal."""
    if df is None or df.empty:
        print("No length correlation data available.")
        return
    
    # Create a view with only the columns we want to show
    view_df = df[["base_length", "next_length", "pass", "pearson_r", "pass_pearson_r", "n_samples"]].copy()
    
    # Format the length columns to show as "2k" instead of 2000
    view_df["base_length"] = view_df["base_length"].apply(lambda x: f"{int(x/1000)}k")
    view_df["next_length"] = view_df["next_length"].apply(lambda x: f"{int(x/1000)}k")
    
    # Format the correlation columns to show rounded values
    view_df["pearson_r"] = view_df["pearson_r"].apply(lambda x: f"{x:.3f}")
    
    # Create table for all length correlations
    print(f"\nCorrelations between edit_score_{weight} at base length and scores at next length:")
    print(tabulate(view_df, headers="keys", tablefmt="grid"))
    
    # Group by base_length and next_length and show average correlations
    grouped = df.groupby(["base_length", "next_length"]).agg({
        "pearson_r": "mean",
        "n_samples": "sum"
    }).reset_index()
    
    # Format the length columns
    grouped["base_length"] = grouped["base_length"].apply(lambda x: f"{int(x/1000)}k")
    grouped["next_length"] = grouped["next_length"].apply(lambda x: f"{int(x/1000)}k")
    
    # Format the correlation columns
    grouped["pearson_r"] = grouped["pearson_r"].apply(lambda x: f"{x:.3f}")
    
    print(f"\nAverage correlations by length pair:")
    print(tabulate(grouped, headers=["Base Length", "Next Length", "Avg Pearson r", "Avg Pass Pearson r", "Total Samples"], 
                  tablefmt="grid"))

def main():
    # Configuration - modify these parameters as needed
    base_dir = "path/to/analysis/files"  # Replace with your actual directory
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
    print(f"\nAnalyzing correlations with edit_score weight {weight}...")
    pass_results, length_results = analyze_next_level_correlations(data_dict, weight=weight)
    
    # Save results to CSV
    if pass_results is not None:
        # pass_results.to_csv(os.path.join(output_dir, f"next_pass_correlations_weight{weight}.csv"), index=False)
        print(f"Pass correlation results saved to CSV ({len(pass_results)} pairs analyzed)")
        # Display correlation tables
        display_pass_correlations_table(pass_results, weight)
    else:
        print("No valid pass correlation pairs found")
    
    if length_results is not None:
        # length_results.to_csv(os.path.join(output_dir, f"next_length_correlations_weight{weight}.csv"), index=False)
        print(f"Length correlation results saved to CSV ({len(length_results)} pairs analyzed)")
        # Display correlation tables
        display_length_correlations_table(length_results, weight)
    else:
        print("No valid length correlation pairs found")
    
    # Print summary of findings
    print("\nSummary of findings:")
    
    if pass_results is not None and not pass_results.empty:
        avg_by_base = pass_results.groupby("base_pass")["pearson_r"].mean()
        best_base_pass = avg_by_base.idxmax()
        print(f"- Best base pass for prediction: pass {best_base_pass} (avg r={avg_by_base[best_base_pass]:.3f})")
        
        best_pair = pass_results.loc[pass_results["pearson_r"].idxmax()]
        print(f"- Strongest pass correlation: pass {int(best_pair.base_pass)} → pass {int(best_pair.next_pass)} "
              f"at length {int(best_pair.length)/1000}k (r={best_pair.pearson_r:.3f})")
    
    if length_results is not None and not length_results.empty:
        avg_by_base = length_results.groupby("base_length")["pearson_r"].mean()
        best_base_length = avg_by_base.idxmax()
        print(f"- Best base length for prediction: {int(best_base_length)/1000}k (avg r={avg_by_base[best_base_length]:.3f})")
        
        best_pair = length_results.loc[length_results["pearson_r"].idxmax()]
        print(f"- Strongest length correlation: {int(best_pair.base_length)/1000}k → {int(best_pair.next_length)/1000}k "
              f"at pass {int(best_pair['pass'])} (r={best_pair.pearson_r:.3f})")
    
    print("\nAnalysis complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main()