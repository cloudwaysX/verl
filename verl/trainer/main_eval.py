# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
import os
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import math, gsm8k
import pandas as pd
import numpy as np


def select_reward_fn(data_source, usedeepscaler=False):
    if data_source == 'lighteval/MATH':
        return math.compute_score
    elif data_source == "DEEPSCALER" or usedeepscaler:
        # To align with the compute score function, we need to wrap the reward function
        def customized_compute_score(solution_str, ground_truth, extra_info=None):
            from deepscaler.rewards.math_reward import deepscaler_reward_fn
            res = deepscaler_reward_fn("<think>"+solution_str, ground_truth)
            
            if isinstance(res, (int, float, bool)):
                return float(res)
            else:
                return float(res[0])
        return customized_compute_score
    else:
        raise NotImplementedError


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    if config.data.get("edit_response_key", None):
        edit_responses = dataset[config.data.edit_response_key]
    else:
        edit_responses = None
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    
    # Get n_pass from config or default to using all responses
    n_pass = config.data.n_pass
    # Get max_response_length from config
    max_response_length = config.data.max_response_length  # Default value if not specified

    # Define edit weights
    edit_weights = [0.8, 0.4, 0.2]
    
    # Initialize result lists
    passes = []
    edit_passes = []
    mean_scores = []
    mean_edit_scores = []
    
    # New metrics
    clip_ratios = []
    score_variances = []
    all_score_variances = []  # Combined variance of original and weighted scores
    weighted_edit_scores = {w: [] for w in edit_weights}
    weighted_edit_passes = {w: [] for w in edit_weights}

    total = len(dataset)
    num_response = responses[0].shape[0]
    
    # Limit number of responses if n_pass is specified
    if n_pass is not None and n_pass < num_response:
        effective_num_response = n_pass
    else:
        effective_num_response = num_response

    for i in range(total):
        response_lst = responses[i][:effective_num_response]  # Apply n_pass limit
        edit_response_lst = edit_responses[i][:effective_num_response] if edit_responses is not None else None
        data_source = data_sources[i]
        # select reward score based on data_source
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source, usedeepscaler=True)
        ground_truth = reward_data['ground_truth']
        
        # Calculate clip ratio based on response length vs max_response_length
        response_lengths = [len(r) for r in response_lst]
        clipped_count = sum(1 for length in response_lengths if length >= max_response_length)
        clip_ratio = clipped_count / effective_num_response
        clip_ratios.append(clip_ratio)
        
        # Score calculations
        score_lst = []
        edit_score_lst = []
        weighted_edit_score_lists = {w: [] for w in edit_weights}
        
        for j, r in enumerate(response_lst):
            score = reward_fn(r, ground_truth)
            score_lst.append(score)
            
            if edit_response_lst is not None:
                edit_response = edit_response_lst[j]
                edit_score = reward_fn(edit_response, ground_truth)
                edit_score_lst.append(edit_score)
                
                # Calculate weighted edit scores for different weights - CORRECTED FORMULA
                for weight in edit_weights:
                    weighted_score = edit_score * weight
                    weighted_edit_score_lists[weight].append(weighted_score)
            else:
                # If no edit, use the original score
                edit_score_lst.append(score)
                for weight in edit_weights:
                    weighted_edit_score_lists[weight].append(score * weight)

        # Calculate metrics
        max_score = np.max(score_lst)
        max_edit_score = np.max(edit_score_lst)
        mean_score = np.mean(score_lst)
        mean_edit_score = np.mean(edit_score_lst)
        score_variance = np.var(score_lst)
        
        # Store metrics
        passes.append(int(max_score == 1))
        edit_passes.append(int(max_edit_score == 1))
        mean_scores.append(mean_score)
        mean_edit_scores.append(mean_edit_score)
        score_variances.append(score_variance)
        
        # Store weighted edit metrics
        for weight in edit_weights:
            weighted_scores = weighted_edit_score_lists[weight]
            weighted_edit_scores[weight].append(np.mean(weighted_scores))
            weighted_edit_passes[weight].append(int(np.max(weighted_scores) == 1))
        
        # Calculate combined variance of original and all weighted scores
        all_scores = score_lst.copy()
        for weight in edit_weights:
            all_scores.extend(weighted_edit_score_lists[weight])
        all_score_variance = np.var(all_scores)
        all_score_variances.append(all_score_variance)

    # Compute the correlation with difficulty
    if config.data.get("difficulty_key", None):
        difficulties = [item[config.data.difficulty_key] for item in dataset["extra_info"]]
        
        # Create results DataFrame
        results_dict = {
            'difficulty': difficulties,
            'mean_score': mean_scores,
            'mean_edit_score': mean_edit_scores,
            'pass': passes,
            'edit_pass': edit_passes,
            'score_variance': score_variances,
            'all_score_variance': all_score_variances,  # Combined variance
            'clip_ratio': clip_ratios
        }
        
        # Add weighted edit metrics to results
        for weight in edit_weights:
            results_dict[f'mean_edit_score_{weight}'] = weighted_edit_scores[weight]
            results_dict[f'edit_pass_{weight}'] = weighted_edit_passes[weight]
        
        results_df = pd.DataFrame(results_dict)
        
        # Count None values in difficulty
        none_count = results_df['difficulty'].isna().sum()
        print(f'Number of None values in difficulty: {none_count}')
        
        # Calculate metrics for all samples
        print(f"\nOverall metrics (using first {effective_num_response} of {num_response} generations):")
        print(f'pass@{effective_num_response}: {results_df["pass"].mean():.4f}')
        print(f'edit_pass@{effective_num_response}: {results_df["edit_pass"].mean():.4f}')
        print(f'mean_score: {results_df["mean_score"].mean():.4f}')
        print(f'mean_edit_score: {results_df["mean_edit_score"].mean():.4f}')
        print(f'mean_score_variance: {results_df["score_variance"].mean():.4f}')
        print(f'mean_all_score_variance: {results_df["all_score_variance"].mean():.4f}')
        print(f'mean_clip_ratio: {results_df["clip_ratio"].mean():.4f}')
        
        # Print weighted edit metrics
        for weight in edit_weights:
            print(f'mean_edit_score_{weight}: {results_df[f"mean_edit_score_{weight}"].mean():.4f}')
            print(f'edit_pass_{weight}@{effective_num_response}: {results_df[f"edit_pass_{weight}"].mean():.4f}')
        
        # Calculate metrics for samples with None difficulty
        if none_count > 0:
            none_df = results_df[results_df['difficulty'].isna()]
            print("\nMetrics for samples with None difficulty:")
            print(f'pass@{effective_num_response}: {none_df["pass"].mean():.4f}')
            print(f'edit_pass@{effective_num_response}: {none_df["edit_pass"].mean():.4f}')
            print(f'mean_score: {none_df["mean_score"].mean():.4f}')
            print(f'mean_edit_score: {none_df["mean_edit_score"].mean():.4f}')
            
            # Print weighted edit metrics for None difficulty
            for weight in edit_weights:
                print(f'mean_edit_score_{weight}: {none_df[f"mean_edit_score_{weight}"].mean():.4f}')
                print(f'edit_pass_{weight}@{effective_num_response}: {none_df[f"edit_pass_{weight}"].mean():.4f}')
        
        # Drop rows with None difficulty for correlation analysis
        valid_df = results_df.dropna(subset=['difficulty'])
        print(f'\nSamples with valid difficulty: {len(valid_df)}/{total}')
        
        # Calculate metrics for samples with valid difficulty
        print("\nMetrics for samples with valid difficulty:")
        print(f'pass@{effective_num_response}: {valid_df["pass"].mean():.4f}')
        print(f'edit_pass@{effective_num_response}: {valid_df["edit_pass"].mean():.4f}')
        print(f'mean_score: {valid_df["mean_score"].mean():.4f}')
        print(f'mean_edit_score: {valid_df["mean_edit_score"].mean():.4f}')
        print(f'mean_all_score_variance: {valid_df["all_score_variance"].mean():.4f}')
        
        # Print weighted edit metrics for valid difficulty
        for weight in edit_weights:
            print(f'mean_edit_score_{weight}: {valid_df[f"mean_edit_score_{weight}"].mean():.4f}')
            print(f'edit_pass_{weight}@{effective_num_response}: {valid_df[f"edit_pass_{weight}"].mean():.4f}')
        
        # Compute correlations
        if len(valid_df) > 1:
            print("\nCorrelations with difficulty:")
            score_columns = ['mean_score', 'mean_edit_score', 'pass', 'edit_pass', 
                             'score_variance', 'all_score_variance', 'clip_ratio']
            
            # Add weighted edit score columns
            for weight in edit_weights:
                score_columns.append(f'mean_edit_score_{weight}')
                score_columns.append(f'edit_pass_{weight}')
                
            for score_column in score_columns:
                correlation = valid_df['difficulty'].corr(valid_df[score_column])
                print(f'Correlation between difficulty and {score_column}: {correlation:.4f}')
        else:
            print("Not enough valid difficulty values to calculate correlation")
            
        # Optional: Save results
        if config.get("output_dir", None):
            outfile = os.path.join(config.output_dir, "analysis.csv")
            results_df.to_csv(outfile, index=False)
            print(f'Results saved to {outfile}')
    else:
        # If no difficulty key, just print overall metrics
        print(f"\nOverall metrics (using first {effective_num_response} of {num_response} generations):")
        print(f'pass@{effective_num_response}: {np.mean(passes):.4f}')
        print(f'edit_pass@{effective_num_response}: {np.mean(edit_passes):.4f}')
        print(f'mean_score: {np.mean(mean_scores):.4f}')
        print(f'mean_edit_score: {np.mean(mean_edit_scores):.4f}')
        print(f'mean_score_variance: {np.mean(score_variances):.4f}')
        print(f'mean_all_score_variance: {np.mean(all_score_variances):.4f}')
        print(f'mean_clip_ratio: {np.mean(clip_ratios):.4f}')
        
        # Print weighted edit metrics
        for weight in edit_weights:
            print(f'mean_edit_score_{weight}: {np.mean(weighted_edit_scores[weight]):.4f}')
            print(f'edit_pass_{weight}@{effective_num_response}: {np.mean(weighted_edit_passes[weight]):.4f}')


if __name__ == '__main__':
    main()