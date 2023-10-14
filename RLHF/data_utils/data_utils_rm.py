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

from dataclasses import dataclass
import json
import logging
from typing import Callable, Optional, Dict, Sequence, List, Tuple

import random

import einops
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

from PIL import Image
import copy
import os

import data_utils.common_utils as utils
from data_utils.common_utils import preprocess, preprocess_multimodal


logger = logging.getLogger(__name__)


def preprocess_for_reward_modeling(
    data: Sequence[dict],
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor: Optional[Callable] = None,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    use_data_frame: bool = True,
    verbose: bool = True,
    has_image: bool = True,
    mask_target: bool = False,
    reward_model_prompt: Optional[str] = None,
    image_to_caption_mapping: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, torch.Tensor]:
    if use_data_frame:
        df = pd.DataFrame(data)
        if df_postprocessor is not None:
            df = df_postprocessor(df)
        list_dict_data = df.to_dict(orient="records")
    else:
        list_dict_data = data

    index_0, index_1 = tuple(
        torch.full(
            size=(len(list_dict_data), 1), fill_value=fill_value, dtype=torch.long
        )
        for fill_value in (0, 1)
    )

    def _get_numeric_preference(example: dict):
        # 1 vs 2 is stored in table, but for modeling we use 0 vs 1; remap here.
        return {1: 0, 2: 1}[example["preference"]]

    choice = torch.tensor(
        [[_get_numeric_preference(dict_data)] for dict_data in list_dict_data]
    )

    def _get_text(example: dict, output_key: str):
        # HACK(sheng):, hack for V1, LLaMA2
        _s = copy.deepcopy(example["conversations"])

        if isinstance(_s[-1], list):
            assert len(_s) == 1
            _s = _s[-1]

        image_captions = None
        if "image" in example:
            image = example["image"]
            if image_to_caption_mapping is not None:
                image_captions = image_to_caption_mapping[image]
                random.shuffle(image_captions)

        assert _s[-1]["from"] == "gpt"
        _s[-1]["value"] = example[output_key]["value"]
        return preprocess(
            [_s],
            tokenizer,
            has_image=has_image,
            mask_target=mask_target,
            query_len=query_len,
            response_len=response_len,
            reward_model_prompt=reward_model_prompt,
            image_captions=[image_captions],
        )

    # TODO(sheng): hack for LLAVA_V1, LLaMA2
    # logger.warning(f"Tokenizing {len(list_dict_data)} pairs...")

    text_list_0, text_list_1 = tuple(
        _get_text(dict_data, key)
        for dict_data in list_dict_data
        for key in ("output_1", "output_2")
    )

    merged_valid = [
        val_1 and val_2
        for val_1, val_2 in zip(text_list_0["validity"], text_list_1["validity"])
    ]

    # "size" (bsz, 2, seq_len)
    input_ids = [
        list(pair)
        for pair in utils.zip_(text_list_0["input_ids"], text_list_1["input_ids"])
    ]
    labels = [
        list(pair) for pair in utils.zip_(text_list_0["labels"], text_list_1["labels"])
    ]

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        index_0=index_0,
        index_1=index_1,
        choice=choice,
        merged_valid=merged_valid,
        metadata=dict(mean_choice=choice.float().mean().item()),
    )

    return packaged_data


class BinaryRewardModelingDataset(Dataset):
    def __init__(
        self,
        data: Sequence[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        query_len: Optional[int] = None,
        response_len: Optional[int] = None,
        use_data_frame: bool = True,
        data_args: Optional[Dict] = None,
    ):
        super(BinaryRewardModelingDataset, self).__init__()
        list_data_dict = json.load(open(data_args.dataset_path, "r"))
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        self.query_len = query_len
        self.response_len = response_len
        self.use_data_frame = use_data_frame

        self.reward_model_prompt = None
        if data_args.reward_prompt_file is not None:
            with open(data_args.reward_prompt_file, "r") as f:
                self.reward_model_prompt = " " + f.read().strip()

        self.image_to_caption_mapping = None
        if data_args.image_to_caption_file is not None:
            with open(data_args.image_to_caption_file, "r") as f:
                self.image_to_caption_mapping = json.load(f)

    def __len__(self):
        # return len(self.input_ids)
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            _sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            _sources = copy.deepcopy([e["conversations"] for e in sources])

        sources_ = copy.deepcopy(sources)
        sources_[0]["conversations"] = _sources

        data_dict = preprocess_for_reward_modeling(
            sources_,
            tokenizer=self.tokenizer,
            has_image=("image" in self.list_data_dict[i]),
            mask_target=False,
            query_len=self.query_len,
            response_len=self.response_len,
            use_data_frame=self.use_data_frame,
            reward_model_prompt=self.reward_model_prompt,
            image_to_caption_mapping=self.image_to_caption_mapping,
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                choice=data_dict["choice"][0],
                index_0=data_dict["index_0"][0],
                index_1=data_dict["index_1"][0],
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image.to(torch.bfloat16)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        return data_dict


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


@dataclass
class DataCollatorForBinaryRewardModelingDataset(object):
    """
    This collation assumes data preprocessing converts text into *padded* tensors of the same length.
    For autoregressive models like OPT and GPT2, `input_ids` alone is sufficient to produce the rewards.
    For enc-dec models like T5, we need `labels`.

    `input_ids` and `labels` are tensors of size (bsz, num_candidates, max_seq_len), i.e., each batch instance has
    `num_candidates` generations/completions.
    `index_0` and `index_1` are tensors of size (bsz, num_pairs), and are used to index into `input_ids` and
    `labels` to find the first and second sequences in the pair.
    `choice` is a binary int/long tensor of size (bsz, num_pairs) indicating which sequence in the pair is better,
    i.e., 0 means the first sequence is preferred, and 1 means otherwise.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        input_ids = [seq for instance in instances for seq in instance[key]]  # Flatten.
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=len(instances[0][key]),
        )
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        index_0, index_1, choice = tuple(
            torch.stack([instance[key] for instance in instances])
            for key in ("index_0", "index_1", "choice")
        )
        input_ids = self._left_pad_helper(instances, "input_ids")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            index_0=index_0,
            index_1=index_1,
            choice=choice,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    if data_args.dataset_path.endswith("json"):
        train_preference = load_dataset("json", data_files=data_args.dataset_path)[
            "train"
        ]
        use_data_frame = False
    else:
        raise ValueError(
            f"Unsupported dataset_path: {data_args.dataset_path}."
            "Only json datasets are supported."
        )

    train_dataset = BinaryRewardModelingDataset(
        data=train_preference,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
        response_len=training_args.response_len,
        use_data_frame=use_data_frame,
        data_args=data_args,
    )

    if (
        data_args.dataset_path == data_args.eval_dataset_path
        and data_args.dataset_name == data_args.eval_dataset_name
    ):
        train_dataset, eval_dataset = split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=data_args.eval_size,
            seed=training_args.seed,
        )

    else:
        raise NotImplementedError

    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
