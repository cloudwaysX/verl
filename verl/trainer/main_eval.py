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


def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        return math.compute_score
    elif datasource="DEEPSCALER":
        # To align with the compute score function, we need to wrap the reward function
        def customized_compute_score(solution_str, ground_truth, extra_info=None):
            from deepscaler.rewards.math_reward import deepscaler_reward_fn
            res = deepscaler_reward_fn(solution_str, ground_truth)
            
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
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        edit_score_lst = []
        for i, r in enumerate(response_lst):
            score = reward_fn(r, ground_truth)
            score_lst.append(score)
            if score < 1 and edit_response_lst is not None:
                edit_response = edit_response_lst[i]
                edit_score = reward_fn(edit_response, ground_truth)
                if edit_score*0.5 > score:
                    score = edit_score*0.5
            edit_score_lst.append(score)
 


        max_score = np.max(score_lst)
        max_edit_score = np.max(edit_score_lst)
        mean_score = np.mean(score_lst)
        mean_edit_score = np.mean(edit_score_lst)

        passes.append(int(max_score == 1))
        edit_passes.append(int(max_edit_score == 1))
        mean_scores.append(mean_score)
        mean_edit_scores.append(mean_edit_score)

    print(f'pass@{num_response}: {sum(passes) / total}')
    print(f'edit_pass@{num_response}): {sum(edit_passes) / total}')
    print(f'mean_score: {sum(mean_scores) / total}')
    print(f'mean_edit_score: {sum(mean_edit_scores) / total}')
    
    # Compute the correlation with difficulty
    if config.data.get("difficulty_key",None):
        difficulty = dataset["extra_info"][config.data.difficulty_key].tolist()
        # Can you complete this?
        mean_correlation = np.corrcoef(difficulty, mean_scores)[0, 1]
        print(f'correlation between difficulty and mean_score: {mean_correlation}')
        mean_correlation = np.corrcoef(difficulty, mean_edit_scores)[0, 1]
        print(f'correlation between difficulty and mean_edit_score: {mean_correlation}')
        pass_correlation = np.corrcoef(difficulty, passes / total)[0, 1]
        print(f'correlation between difficulty and pass: {pass_correlation}')
        edit_pass_correlation = np.corrcoef(difficulty, edit_passes / total)[0, 1]
        print(f'correlation between difficulty and edit_pass: {edit_pass_correlation}')
        


if __name__ == '__main__':
    main()
