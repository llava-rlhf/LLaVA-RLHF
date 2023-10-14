# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
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

import dataclasses
from typing import Callable, Dict, Optional, List, Sequence

import logging
import pandas as pd

import torch
from torch.utils.data import Dataset

import transformers
import datasets

import data_utils.common_utils as utils

from PIL import Image
import copy
import os
import tqdm

import data_utils.common_utils as utils
from data_utils.common_utils import preprocess, preprocess_multimodal

logger = logging.getLogger(__name__)


class QueryResponseDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
        query_len: int,
        df_postprocessor: Optional[Callable] = None,
        data_args: Optional[Dict] = None,
    ):
        self.data_args = data_args
        super(QueryResponseDataset, self).__init__()

        if df_postprocessor is not None:
            df = df_postprocessor(df)
        list_dict_data = df.to_dict(orient="records")

        _s = copy.deepcopy([ex["conversations"] for ex in list_dict_data])
        _s = preprocess_multimodal(_s, data_args)

        _s = [__s[:2] for __s in _s]

        for __s in _s:
            assert __s[-1]["from"] == "gpt", f"{__s}"
            __s[-1]["value"] = "\n"

        queries = [
            preprocess(
                [__s],
                tokenizer,
                has_image=True,
                mask_target=False,
                query_len=query_len,
            )["input_ids"]
            for __s in tqdm.tqdm(_s)
        ]

        queries = [
            torch.tensor(query, dtype=torch.long).view(-1)[:-3] for query in queries
        ]

        filtered_queries = []

        for query in queries:
            if len(query) <= query_len:
                filtered_queries.append(query)

        max_query_len = max(len(query) for query in filtered_queries)
        logger.warning(f"Max query length: {max_query_len}")

        logger.warning(
            f"Filtered out {len(queries) - len(filtered_queries)} instances out of {len(queries)} that "
            f"exceed length limit. These examples are not used for training, but will still be used in evaluation. "
        )

        queries = torch.stack(
            [
                utils.left_pad(
                    query, target_size=(query_len,), value=tokenizer.pad_token_id
                )
                for query in filtered_queries
            ]
        )

        self.queries = queries
        self.query_attn_masks = queries.ne(tokenizer.pad_token_id).long()

        # Auxiliary data.
        self.list_dict_data = list_dict_data

    def __getitem__(self, idx):
        return_dict = dict(
            queries=self.queries[idx],
            query_attn_masks=self.query_attn_masks[idx],
        )

        image_file = self.list_dict_data[idx]["image"]
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        reward_processor = self.data_args.reward_image_processor

        if "caption_type" in self.list_dict_data[idx]:
            caption_type = self.list_dict_data[idx]["caption_type"]
        else:
            caption_type = 1

        if "length_bonus" in self.list_dict_data[idx]:
            length_bonus = self.list_dict_data[idx]["length_bonus"]
        else:
            length_bonus = 1.0

        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except:
            raise ValueError(f"Error loading image {image_file} for index {idx}")

        original_image = image

        if self.data_args.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            reward_image = expand2square(
                original_image, tuple(int(x * 255) for x in reward_processor.image_mean)
            )
            reward_image = reward_processor.preprocess(
                reward_image, return_tensors="pt"
            )["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            reward_image = reward_processor.preprocess(
                original_image, return_tensors="pt"
            )["pixel_values"][0]

        return_dict["images"] = image
        return_dict["reward_images"] = reward_image
        return_dict["image_file_ids"] = torch.tensor(
            int(image_file.split(".")[0]), dtype=torch.long
        )
        return_dict["caption_types"] = torch.tensor(caption_type, dtype=torch.long)
        return_dict["length_bonus_multiplier"] = torch.tensor(
            length_bonus, dtype=torch.float
        )

        return return_dict

    def __len__(self):
        return len(self.queries)


@dataclasses.dataclass
class DataCollatorForQueryResponseDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([instance[key] for instance in instances])
            for key in instances[0].keys()
        }


def make_rl_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    if data_args.dataset_path.endswith("json"):
        train_instructions = datasets.load_dataset(
            "json", data_files=data_args.dataset_path
        )
    else:
        train_instructions = datasets.load_dataset(
            data_args.dataset_path, data_args.dataset_name
        )
    train_df = pd.concat(
        [pd.DataFrame(train_instructions[split]) for split in data_args.train_splits]
    )

    train_dataset = QueryResponseDataset(
        df=train_df,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
        data_args=data_args,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForQueryResponseDataset(),
    )
