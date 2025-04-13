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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, edit_weight=1, max_response_length=99999) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.edit_weight = edit_weight
        self.max_response_length = max_response_length

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # When compute the reward, always use the initial response
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            initial_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Always try with original responses first
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if "edit_responses" not in data.batch:
                # If edit_responses does not exist, that means we may in the overwrite mode.
                # In this case, we use the score weight by examining the length of the response
                if initial_response_length > self.max_response_length:
                    # print("DEBUG use edit weight")
                    score = score * self.edit_weight
            
            # If the score is less than 1 and edit_responses exists, try with edited response
            if score < 1 and 'edit_responses' in data.batch and self.edit_weight > 0:
                assert self.edit_weight is not None, 'edit_weight is not set while edit_responses exists'
                # Try with edited responses
                edit_response_ids = data_item.batch['edit_responses']
                edit_valid_response_length = data_item.batch['edit_attention_mask'][prompt_length:].sum()
                edit_valid_response_ids = edit_response_ids[:edit_valid_response_length]
                
                # decode with edited response
                edit_sequences = torch.cat((valid_prompt_ids, edit_valid_response_ids))
                edit_sequences_str = self.tokenizer.decode(edit_sequences)
                
                edit_score = self.compute_score(
                    data_source=data_source,
                    solution_str=edit_sequences_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                
                # Apply the edit weight and compare with original score
                weighted_edit_score = edit_score * self.edit_weight
                
                # If weighted edit score is better than original score, use it
                if weighted_edit_score > score:
                    score = weighted_edit_score 
                    sequences_str = edit_sequences_str
            
            # When compute the reward, always use the initial response length for positioning the reward
            reward_tensor[i, initial_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor
