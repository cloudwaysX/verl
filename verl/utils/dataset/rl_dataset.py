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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.preselect import selection_for_math_difficulty, selection_for_mathamc_difficulty
from verl.trainer.ppo.preselect import selection_for_deepscaler_difficulty
from verl.trainer.ppo.preselect import selection_for_openthoughts_difficulty, balance_dataset_by_ability
from verl.trainer.ppo.oed import coreset_selection, reverse_coreset_selection, redant_selection


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 train_ratio=1,
                 train_ratio_seed=None,
                 embedding_path: str = None,
                 oed: Optional[str] = None,
                 oed_save_path: Optional[str] = None):
        """
        oed:
            balance_by_ability: balance the dataset by ability
            math_difficulty: select hard examples for math
            mathamc_difficulty: select hard examples for mathamc
            deepscaler_difficulty38: select hard examples for deepscaler
            openthoughts_difficulty4: select hard examples for openthoughts
            coreset: coreset selection
        embedding_path: path to embedding use for some oed strategies
        """
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.use_original_id = False

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize(
                train_ratio, 
                train_ratio_seed, 
                embedding_path,
                oed,
                oed_save_path)
        self._set_all_prompt_ids()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self, 
                                 train_ratio, 
                                 train_ratio_seed=None, 
                                 embedding_path=None,
                                 oed="random",
                                 oed_save_path=None):
        dfs = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dfs.append(dataframe)
        dfs = pd.concat(dfs)
        
        # optionally load embeddings (must align 1:1 with df rows)
        if oed in ["coreset", "reverse_coreset"]:
            if embedding_path:
                embeddings = np.load(embedding_path)
                assert embeddings.shape[0] == len(dfs), (
                    f"Embeddings length {embeddings.shape[0]} != #samples {len(dfs)}"
                )
            else:
                raise ValueError("No embedding path provided for coreset oed")
        else:
            embeddings = None

        # Filter out prompts that are too long
        mask = dfs.apply(
            lambda row: len(
                self.tokenizer.apply_chat_template(
                    row[self.prompt_key],
                    add_generation_prompt=True
                )
            ) <= self.max_prompt_length,
            axis=1
        ).to_numpy()
        dfs = dfs.loc[mask].reset_index(drop=True)
        if embeddings is not None:
            embeddings = embeddings[mask]
            
        self.dataframe = dfs
        self.embeddings = embeddings
        
        # decide the training budget
        size = int(len(self.dataframe)*train_ratio)
            
        if oed in ["balance_by_ability"]:
            self.dataframe = balance_dataset_by_ability(self.dataframe, size, train_ratio_seed)
        elif oed in ['math_difficulty']:
            self.dataframe = selection_for_math_difficulty(self.dataframe)
        elif oed in ["mathamc_difficulty"]:
            self.dataframe = selection_for_mathamc_difficulty(self.dataframe)
        elif oed in ["deepscaler_difficulty38"]:
            self.dataframe = selection_for_deepscaler_difficulty(self.dataframe)
        elif oed in ["openthoughts_difficulty4"]:
            self.dataframe = selection_for_openthoughts_difficulty(self.dataframe)
        elif oed in ["coreset"]:
            idxs = coreset_selection(embeddings, size, oed_save_path, train_ratio_seed)
            self.dataframe = self.dataframe.iloc[idxs]
        elif oed in ["reverse_coreset", "reverse_coreset_initsize100"]:
            if oed == "reverse_coreset_initsize100":
                initial_seedsamples_size = 100
            else:
                initial_seedsamples_size = 1
            idxs = reverse_coreset_selection(embeddings, 
                                             size, 
                                             oed_save_path, 
                                             train_ratio_seed,
                                             initial_seedsamples_size)
            self.dataframe = self.dataframe.iloc[idxs]
        elif oed in ["redant"]:
            # This one does not use filtered embeddings, so we need extra process....
            # TODO: A better way to do this is to make redant_selection work with filtered embeddings
            idxs = redant_selection(size, oed_save_path, train_ratio_seed)
            # Store the original indices that were kept
            original_indices_kept = np.where(mask)[0]
            print(f"Original indices kept has len {len(original_indices_kept)}")
            # Create a mapping from original index to new index for kept items
            # Example: if original_indices_kept is [0, 1, 3, 4, 5]
            # Then original index 0 maps to new index 0
            # Then original index 1 maps to new index 1
            # Then original index 3 maps to new index 2
            # etc.
            original_to_new_index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(original_indices_kept)}
            new_selected_idx = []
            for original_idx in idxs:
                # Check if this original index was kept by the mask
                if original_idx in original_to_new_index_map:
                    # If kept, find its new index and add it to the new list
                    new_selected_idx.append(original_to_new_index_map[original_idx])
            self.dataframe = self.dataframe.iloc[new_selected_idx]
        elif oed in ["random"]:
            if train_ratio_seed is not None:
                np.random.seed(train_ratio_seed)
                self.dataframe = self.dataframe.sample(frac=1, random_state=train_ratio_seed).reset_index(drop=True)
            self.dataframe = self.dataframe.head(size)

        print(f"The len of final dataset is {len(self.dataframe)}")

    def resume_dataset_state(self,train_ratio=1):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize(train_ratio)
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)
    
    def _set_all_prompt_ids(self):
        # If 'extra_info' exists in the DataFrame, try to extract the 'index' from it.
        if 'extra_info' in self.dataframe.columns:
            # Apply a function to extract the index from extra_info if present.
            def extract_index(row):
                # row is a pandas Series
                extra = row.get('extra_info', {})
                if isinstance(extra, dict) and 'index' in extra:
                    return extra['index']
                else:
                    return None  # or you could return row.name as fallback

            indices = self.dataframe.apply(extract_index, axis=1)
            # If all rows have a valid index from extra_info, use them.
            if indices.notnull().all() and indices.is_unique:
                out = indices.tolist()
                self.use_original_id=True
            else:
                # Fallback: use the DataFrame's inherent index.
                out = self.dataframe.index.tolist()
        else:
            # Otherwise, just use the DataFrame's index.
            out = self.dataframe.index.tolist()
            
        self.rawindex2rowindex = {v: k for k, v in enumerate(out)}
        self.rowindex2rawindex = out  
        return out
    
    def get_all_topics(self):
        return set(self.dataframe["ability"])
    
    def get_all_prompt_ids(self):
        #legacy, pretty much similar to below
        return list(self.rawindex2rowindex.keys())
    def get_all_prompt_ids_inorder(self):
        return self.rowindex2rawindex

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # Check if there is an index inside the extra_info field.
        # If it exists, use it; otherwise, assign the DataFrame's index.
        if self.use_original_id:
            row_dict["index"] = row_dict["extra_info"]["index"]
        else:
            row_dict["index"] = self.dataframe.index[item]
            
        if "extra_info" in row_dict and "difficulty" in row_dict["extra_info"]:
            row_dict["difficulty"] = row_dict["extra_info"]["difficulty"]

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
