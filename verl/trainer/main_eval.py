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
    if config.data.get("edit_response_key",None):
        edit_responses = dataset[config.data.edit_response_key]
    else:
        edit_responses = None
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = []
    edit_passes = []
    mean_scores = []
    mean_edit_scores = []

    total = len(dataset)
    num_response = responses[0].shape[0]

    for i in range(total):
        response_lst = responses[i]
        edit_response_lst = edit_responses[i] if edit_responses is not None else None
        data_source = data_sources[i]
        # select reward score based on data_source
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source, usedeepscaler=True)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        edit_score_lst = []
        for i, r in enumerate(response_lst):
            score = reward_fn(r, ground_truth)
            score_lst.append(score)
            if score < 1 and edit_response_lst is not None:
                edit_response = edit_response_lst[i]
                edit_score = reward_fn(edit_response, ground_truth)
                if edit_score > score:
                    score = edit_score
            edit_score_lst.append(score)
 


        max_score = np.max(score_lst)
        max_edit_score = np.max(edit_score_lst)
        mean_score = np.mean(score_lst)
        mean_edit_score = np.mean(edit_score_lst)

        passes.append(int(max_score == 1))
        edit_passes.append(int(max_edit_score == 1))
        mean_scores.append(mean_score)
        mean_edit_scores.append(mean_edit_score)


    # Compute the correlation with difficulty
    if config.data.get("difficulty_key",None):
        difficulties = [item[config.data.difficulty_key] for item in dataset["extra_info"]]
        results_df = pd.DataFrame({
            'difficulty': difficulties,
            'mean_score': mean_scores,
            'mean_edit_score': mean_edit_scores,
            'pass': passes,
            'edit_pass': edit_passes
        })
        
        # Count None values in difficulty
        none_count = results_df['difficulty'].isna().sum()
        print(f'Number of None values in difficulty: {none_count}')
        
        # Calculate metrics for all samples
        print("\nOverall metrics:")
        print(f'pass@{num_response}: {results_df["pass"].mean():.4f}')
        print(f'edit_pass@{num_response}: {results_df["edit_pass"].mean():.4f}')
        print(f'mean_score: {results_df["mean_score"].mean():.4f}')
        print(f'mean_edit_score: {results_df["mean_edit_score"].mean():.4f}')
        
        # Calculate metrics for samples with None difficulty
        if none_count > 0:
            none_df = results_df[results_df['difficulty'].isna()]
            print("\nMetrics for samples with None difficulty:")
            print(f'pass@{num_response}: {none_df["pass"].mean():.4f}')
            print(f'edit_pass@{num_response}: {none_df["edit_pass"].mean():.4f}')
            print(f'mean_score: {none_df["mean_score"].mean():.4f}')
            print(f'mean_edit_score: {none_df["mean_edit_score"].mean():.4f}')
        
        # Drop rows with None difficulty for correlation analysis
        valid_df = results_df.dropna(subset=['difficulty'])
        print(f'\nSamples with valid difficulty: {len(valid_df)}/{total}')
        
        # Calculate metrics for samples with valid difficulty
        print("\nMetrics for samples with valid difficulty:")
        print(f'pass@{num_response}: {valid_df["pass"].mean():.4f}')
        print(f'edit_pass@{num_response}: {valid_df["edit_pass"].mean():.4f}')
        print(f'mean_score: {valid_df["mean_score"].mean():.4f}')
        print(f'mean_edit_score: {valid_df["mean_edit_score"].mean():.4f}')
        
        # Compute correlations
        if len(valid_df) > 1:
            print("\nCorrelations with difficulty:")
            for score_column in ['mean_score', 'mean_edit_score', 'pass', 'edit_pass']:
                correlation = valid_df['difficulty'].corr(valid_df[score_column])
                print(f'Correlation between difficulty and {score_column}: {correlation:.4f}')
        else:
            print("Not enough valid difficulty values to calculate correlation")
            
        # Optional: Save results
        if config.get("output_dir", None):
            outfile = os.join(config.output_dir, "analysis.csv")
            results_df.to_csv(outfile, index=False)
            print(f'Results saved to {outfile}')


if __name__ == '__main__':
    main()
