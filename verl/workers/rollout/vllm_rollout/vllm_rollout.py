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

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
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

        self.pad_token_id = tokenizer.pad_token_id
        self.finalans_token = tokenizer.encode(
            '\n\n**Final Answer**: ', add_special_tokens=False
        )
        self.thought_delimiter_start = tokenizer.encode(
            THOUGHT_DELIMITER_START, add_special_tokens=False
        )
        self.thought_delimiter_end = tokenizer.encode(
            THOUGHT_DELIMITER_END, add_special_tokens=False
        )

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

    @torch.no_grad()
    def generate_sequences2(self, prompts: DataProto,  **kwargs) -> DataProto:    
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
        
        # =========Second Generation Pass: Generate final answer using COT =========
        # (yifangc): implement this
        if force_append_answer:
            # tokenize the final answer
            finalans_token = self.thought_delimiter_end + self.finalans_token
            # Convert the prompt+initial response to a list of list of token ids
            extend_input_idx_list = []
            for i in range(batch_size):
                striped_seq = _pre_process_extended_inputs(eos_token_id, initial_response[i])
                striped_seq = (
                    idx_list[i // self.config.n] + striped_seq + finalans_token
                )
                extend_input_idx_list.append(striped_seq)

            kwargs2 = {
                'n': 1,
                'max_tokens': 50,
            }
            with self.update_sampling_params(**kwargs2):
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=extend_input_idx_list,
                    use_tqdm=False)
            second_response = output[0].to(idx.device)
            response_list = []
            for i in range(batch_size):
                orig_prompt_len = len(idx_list[i//self.config.n])
                seq = extend_input_idx_list[i][orig_prompt_len:]+second_response[i].tolist()
                response_list.append(torch.tensor(seq, device=idx.device, dtype=idx.dtype))
            response = pad_sequence(response_list, batch_first=True, padding_value=self.pad_token_id)
        else:
            response = initial_response
        max_response_length = self.config.response_length + 50 if force_append_answer else self.config.response_length
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, max_response_length, self.pad_token_id)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
    
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
        if force_append_answer:
            # tokenize the final answer
            finalans_token = self.thought_delimiter_end + self.finalans_token
            # Convert the prompt+initial response to a list of list of token ids
            extend_input_idx_list = []
            for i in range(batch_size):
                striped_seq = _pre_process_extended_inputs(eos_token_id, initial_response[i])
                striped_seq = (
                    idx_list[i // self.config.n] + striped_seq + finalans_token
                )
                extend_input_idx_list.append(striped_seq)

            kwargs2 = {
                'n': 1,
                'max_tokens': MAX_FINAL_ANSWER_LENGTH,
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
                seq = extend_input_idx_list[i][orig_prompt_len:]+second_response[i].tolist()
                response_list.append(torch.tensor(seq, device=idx.device, dtype=idx.dtype))
            
            edit_response = pad_sequence(response_list, batch_first=True, padding_value=self.pad_token_id)
            
            # Make sure edit_response has correct length before calculating attention mask
            if edit_response.shape[1] < self.config.response_length + MAX_FINAL_ANSWER_LENGTH:
                edit_response = pad_sequence_to_length(
                    edit_response, 
                    self.config.response_length + MAX_FINAL_ANSWER_LENGTH, 
                    self.pad_token_id
                )
            
            # Create edit_attention_mask based on edit_response
            edit_attention_mask = get_eos_mask(
                response_id=edit_response, 
                eos_token=eos_token_id, 
                dtype=attention_mask.dtype
            )
            edit_attention_mask = torch.cat([attention_mask, edit_attention_mask], dim=-1)

            batch.update({
                'edit_responses': edit_response,
                'edit_attention_mask': edit_attention_mask,
            })
            
            if self.config.get("generation_mode", False):
                edit_sequence = torch.cat([idx, edit_response], dim=-1)
                batch.update({
                    'edit_input_ids': edit_sequence,
                })

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
