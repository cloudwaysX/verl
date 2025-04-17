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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import re
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from deepscaler.globals import OAI_RM_MODEL, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

MAX_FINAL_ANSWER_LENGTH = 50


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _pre_process_extended_inputs(eos_token_id, extended_input_ids: torch.Tensor):
    # remove the right padding after the initial response
    # print("debug 2", extended_input_ids.shape)
    non_eos_index = extended_input_ids != eos_token_id
    # print("debug 3", non_eos_index.shape)
    if non_eos_index.any():
        last_valid_index = non_eos_index.nonzero(as_tuple=False)[-1].item()
        token_ids_tensor = extended_input_ids[:last_valid_index+1]
    else:
        token_ids_tensor = torch.tensor([], dtype=extended_input_ids.dtype)
    return token_ids_tensor.tolist()
    




class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get('max_num_batched_tokens', 8192))

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
                
        self.append_answer_len = self.config.get('append_answer_len', MAX_FINAL_ANSWER_LENGTH)
        # TODO(yifangc): not consider adding the response length
        
        self.pad_token_id = tokenizer.pad_token_id
        finalans = self.config.get('finalans', '\n\n**Final Answer**: ')
        self.finalans_token = tokenizer.encode(
            finalans: ', add_special_tokens=False
        )
        self.finalrethink_token = tokenizer.encode(
            '\n\n Wait, let\'s verify again: ', add_special_tokens=False
        )
        self.thought_delimiter_start = tokenizer.encode(
            THOUGHT_DELIMITER_START, add_special_tokens=False
        )
        self.thought_delimiter_end = tokenizer.encode(
            THOUGHT_DELIMITER_END, add_special_tokens=False
        )
        self.append_rethink_tokens = self.config.get('append_rethink_tokens', False)
        self.forceans_for_untruncated = self.config.get('forceans_for_untruncated', False)
        assert not (self.forceans_for_untruncated and self.append_rethink_tokens), \
            "forceans_for_untruncated and append_rethink_tokens cannot be both True"

        finalans_token_len = len(self.thought_delimiter_end + self.finalans_token)
        possible_model_len = config.prompt_length + config.response_length + self.append_answer_len + finalans_token_len
        assert model_hf_config.max_position_embeddings >= possible_model_len, \
            "model context length should be greater than total sequence length"
        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else possible_model_len
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    
    
    def generate_sequences(self, prompts: DataProto,  **kwargs) -> DataProto:
        
        force_append_answer = self.config.force_append_answers
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
            
        if prompts.meta_info.get('longer_response', False):
            kwargs["max_tokens"] = self.config.response_length*2

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        initial_response = output[0].to(idx.device)
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        
        # Make sure initial_response has correct length before calculating position IDs and attention masks
        if initial_response.shape[1] < self.config.response_length:
            initial_response = pad_sequence_to_length(
                initial_response, 
                self.config.response_length, 
                self.pad_token_id
            )
        
        # Create initial response attention mask and position IDs
        response_length = initial_response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        initial_response_position_ids = position_ids[:, -1:] + delta_position_id
        initial_response_attention_mask = get_eos_mask(
            response_id=initial_response, 
            eos_token=eos_token_id, 
            dtype=attention_mask.dtype
        )
        
        # Prepare the initial sequence data
        initial_seq = torch.cat([idx, initial_response], dim=-1)
        initial_position_ids = torch.cat([position_ids, initial_response_position_ids], dim=-1)
        initial_attention_mask = torch.cat([attention_mask, initial_response_attention_mask], dim=-1)
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': initial_response,  # Always use initial_response
                'input_ids': initial_seq,  # prompt + initial_response
                'attention_mask': initial_attention_mask,  # Always use initial_attention_mask
                'position_ids': initial_position_ids,  # Always use initial_position_ids
            },
            batch_size=batch_size)
        
        # =========Second Generation Pass: Generate final answer using COT =========
        # If validation, we always append the final answer to maximize results.
        if not prompts.meta_info.get('longer_response', False) and \
           (force_append_answer or (prompts.meta_info.get('validate', False) and self.config.get("use_edit_for_validation", False))):
               
            if prompts.meta_info.get('validate', False):
                append_think_tokens = False
            else:
                append_think_tokens = self.append_rethink_tokens

            # tokenize the final answer
            finalans_token = self.thought_delimiter_end + self.finalans_token
            finalans_token_len = len(finalans_token)
            finalrethink_token = self.thought_delimiter_start + self.finalrethink_token
            finalrethink_token_len = len(finalrethink_token)
            finalappend_positions = []
            # Convert the prompt+initial response to a list of list of token ids
            extend_input_idx_list = []
            for i in range(batch_size):
                striped_seq = _pre_process_extended_inputs(eos_token_id, initial_response[i])
                
                finalappend_positions.append(len(striped_seq))
                if append_think_tokens and len(striped_seq) < self.config.response_length:
                    striped_seq = (
                            idx_list[i // self.config.n] + striped_seq + finalrethink_token
                    )
                else:
                    striped_seq = (
                        idx_list[i // self.config.n] + striped_seq + finalans_token
                    )
                
                extend_input_idx_list.append(striped_seq)

            kwargs2 = {
                'n': 1,
                'max_tokens': self.append_answer_len,
            }
            with self.update_sampling_params(**kwargs2):
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=extend_input_idx_list,
                    use_tqdm=False)
            second_response = output[0].to(idx.device)
            
            # Create the edited response by combining initial response with second response
            response_list = []
            for i in range(batch_size):
                orig_prompt_len = len(idx_list[i//self.config.n])
                if finalappend_positions[i] >= self.config.response_length:
                    seq = extend_input_idx_list[i][orig_prompt_len:]+second_response[i].tolist()
                elif append_think_tokens:
                    seq = extend_input_idx_list[i][orig_prompt_len:]+second_response[i].tolist()
                elif not self.forceans_for_untruncated:
                    seq = extend_input_idx_list[i][orig_prompt_len:-finalans_token_len] + [eos_token_id]
                response_list.append(torch.tensor(seq, device=idx.device, dtype=idx.dtype))
            
            edit_response = pad_sequence(response_list, batch_first=True, padding_value=self.pad_token_id)
            
            # Make sure edit_response has correct length before calculating attention mask
            # print("edit_response.shape: before final padding", edit_response.shape)
            # print("len of finalans_token: ", len(finalans_token))
            if edit_response.shape[1] < self.config.response_length + self.append_answer_len + max(finalans_token_len, finalrethink_token_len):
                edit_response = pad_sequence_to_length(
                    edit_response, 
                    self.config.response_length + self.append_answer_len + max(finalans_token_len, finalrethink_token_len), 
                    self.pad_token_id
                )
            # print("edit_response.shape: after final padding", edit_response.shape)
            
            # Create edit_attention_mask based on edit_response
            edit_attention_mask = get_eos_mask(
                response_id=edit_response, 
                eos_token=eos_token_id, 
                dtype=attention_mask.dtype
            )
            
            # now mask out the finalans_token in each sequence
            logprob_mask = edit_attention_mask.clone()
            for i in range(batch_size):
                mask_start = finalappend_positions[i]
                if mask_start >= self.config.response_length or self.forceans_for_untruncated:
                    mask_end = mask_start + finalans_token_len
                    logprob_mask[i, mask_start:min(mask_end, logprob_mask.size(1))] = 0
                elif append_think_tokens:
                    mask_end = mask_start + finalrethink_token_len
                    logprob_mask[i, mask_start:min(mask_end, logprob_mask.size(1))] = 0

            
            edit_attention_mask = torch.cat([attention_mask, edit_attention_mask], dim=-1)
            logprob_mask = torch.cat([attention_mask, logprob_mask], dim=-1)
            
            edit_sequence = torch.cat([idx, edit_response], dim=-1)
            # create edit response position IDs
            edit_response_length = edit_response.size(1)
            edit_delta_position_id = torch.arange(1, edit_response_length + 1, device=position_ids.device)
            edit_delta_position_id = edit_delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            edit_response_position_ids = position_ids[:, -1:] + edit_delta_position_id
            edit_position_ids = torch.cat([position_ids, edit_response_position_ids], dim=-1)
            
            
            if prompts.meta_info.get('validate', False) and self.config.get("use_edit_for_validation", False):
                force_append_answer = "edit"
            elif force_append_answer == "alternate":
                if prompts.meta_info['epoch'] % 2 == 1:
                    force_append_answer = "edit"
                else:
                    force_append_answer = "overwrite"
            
            if force_append_answer == "edit":
                batch.update({
                    'edit_responses': edit_response,
                    'edit_attention_mask': edit_attention_mask,
                    'edit_input_ids': edit_sequence,
                    'edit_position_ids': edit_position_ids,
                })
            elif force_append_answer == "overwrite":
                batch.update({
                    'responses': edit_response,
                    'attention_mask': edit_attention_mask,
                    'input_ids': edit_sequence,
                    'position_ids': edit_position_ids,
                    "logprob_mask": logprob_mask,
                })
            else:
                raise ValueError(
                    f'force_append_answer must be either "edit" or "overwrite", but got {force_append_answer}'
                )
            

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
    
    
