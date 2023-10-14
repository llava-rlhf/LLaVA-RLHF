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

import copy
import os

from dataclasses import dataclass
from typing import Dict, Sequence
from PIL import Image

import torch
import transformers
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from llava.constants import IGNORE_INDEX

from data_utils.common_utils import preprocess, preprocess_multimodal


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str, pad_token: int):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        input_ids = [instance[key] for instance in instances]  # Flatten.
        try:
            input_ids = pad_sequence_from_left(
                input_ids,
                batch_first=True,
                padding_value=pad_token,
            )
        except:
            raise ValueError(f"Error padding {key} for {input_ids}")
        return input_ids

    def _right_pad_helper(self, instances: Sequence[dict], key: str, pad_token: int):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        input_ids = [instance[key] for instance in instances]  # Flatten.
        try:
            input_ids = pad_sequence_from_right(
                input_ids,
                batch_first=True,
                padding_value=pad_token,
            )
        except:
            raise ValueError(f"Error padding {key} for {input_ids}")
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = self._right_pad_helper(
            instances, "input_ids", self.tokenizer.pad_token_id
        )
        labels = self._right_pad_helper(instances, "labels", IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        assert input_ids.shape == labels.shape
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


class SupervisedVisionLanguageDataset(Dataset):
    def __init__(
        self,
        data_args: Dict,
        hf_dataset: HFDataset,
    ):
        super(SupervisedVisionLanguageDataset).__init__()
        self.data_args = data_args
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sources = self.hf_dataset[idx]

        image = None
        if "image" in sources:
            image_file = sources["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert(
                    "RGB"
                )
            except:
                raise ValueError(f"Error loading image {image_file} for index {idx}")
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

        data_dict = copy.deepcopy(sources)
        if isinstance(idx, int):
            data_dict = dict(
                input_ids=torch.Tensor(data_dict["input_ids"][0]).long(),
                labels=torch.Tensor(data_dict["labels"][0]).long(),
            )
        else:
            raise ValueError(f"Error loading data for index {idx}")

        if image is not None:
            data_dict["image"] = image

        return data_dict


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


def pad_sequence_from_right(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    return padded_sequence


def local_dataset(dataset_name):
    if dataset_name.endswith(".json"):
        full_dataset = load_dataset("json", data_files=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    return full_dataset


def extract_v1_dataset(
    example,
    tokenizer,
    data_args,
    has_image=True,
    mask_target=True,
    query_len=None,
    response_len=None,
):
    _s = copy.deepcopy(example["conversations"])

    _s = preprocess_multimodal([_s], data_args)[0]

    if isinstance(_s[-1], list):
        assert len(_s) == 1
        _s = _s[-1]

    assert _s[-1]["from"] == "gpt"
    return preprocess(
        [_s],
        tokenizer,
        has_image=has_image,
        mask_target=mask_target,
        query_len=query_len,
        response_len=response_len,
    )


def make_sft_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    data_args,
) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """

    def load_data(dataset_name):
        if os.path.exists(dataset_name):
            try:
                args.dataset_format = (
                    args.dataset_format if args.dataset_format else "alpaca"
                )
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not implemented yet."
            )

    def format_dataset(dataset, dataset_format):
        if dataset_format == "v1":
            dataset = dataset.map(
                lambda ex: extract_v1_dataset(
                    ex,
                    tokenizer=tokenizer,
                    data_args=data_args,
                    query_len=args.source_max_len,
                    response_len=args.target_max_len,
                ),
                num_proc=16,
            )
        else:
            raise NotImplementedError

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["image", "input_ids", "labels"]
            ]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`"
            )
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )
    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
    )
    return dict(
        train_dataset=SupervisedVisionLanguageDataset(data_args, train_dataset)
        if args.do_train
        else None,
        eval_dataset=SupervisedVisionLanguageDataset(data_args, eval_dataset)
        if args.do_eval
        else None,
        predict_dataset=SupervisedVisionLanguageDataset(data_args, eval_dataset)
        if args.do_predict
        else None,
        data_collator=data_collator,
    )
