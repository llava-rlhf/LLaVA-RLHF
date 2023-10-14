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

import argparse
import glob
import os
import random
from typing import (
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
    Mapping,
    Any,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from llava import conversation as conversation_lib
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.train.train import DataArguments
from llava.mm_utils import tokenizer_image_token

from data_utils.constants import FACTUAL_PROMPT

Numeric = Union[int, float]


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):  # collections.Mapping fails in py3.10.
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def merge_dict(dicts: Sequence[dict], merge_fn: Callable = lambda *args: args) -> dict:
    """Merge a sequence of dicts (with the same set of keys) into a single dict."""
    if len(dicts) == 0:
        return dict()
    return {key: merge_fn([dict_[key] for dict_ in dicts]) for key in dicts[0].keys()}


def prepare_inputs(
    data: Union[torch.Tensor, Any], device: Union[str, int, torch.device]
) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)(
            {k: prepare_inputs(v, device) for k, v in data.items()}
        )  # noqa
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)  # This can break with deepspeed.
    return data


def compute_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    return -F.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index
    )


def pad(
    inputs: torch.Tensor,
    target_size: Union[torch.Size, Sequence[int]],
    value=0.0,
    left=True,
):
    current_size = inputs.size()
    diffs = tuple(ti - ci for ti, ci in zip_(target_size, current_size))
    pad_params = []
    for diff in diffs:
        pad_params = ([diff, 0] if left else [0, diff]) + pad_params
    res = F.pad(inputs, pad=pad_params, value=value)
    return res


def left_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=True)


def right_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=False)


def manual_seed(args_or_seed: Union[int, argparse.Namespace], fix_cudnn=False):
    if hasattr(args_or_seed, "seed"):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def make_meta_prompts(meta_prompt_pattern: str):
    meta_prompt_files = glob.glob(meta_prompt_pattern)
    print(f"Found {len(meta_prompt_files)} meta prompts: {meta_prompt_files}")

    meta_prompts = []
    for meta_prompt_file in meta_prompt_files:
        with open(meta_prompt_file, "r") as f:
            meta_prompt = f.readlines()
        meta_prompt = "".join(meta_prompt).strip()
        meta_prompts.append(meta_prompt)
    return meta_prompts


class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    mask_target: bool = True,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    mask_target: bool = True,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    reward_model_prompt: Optional[str] = None,
    image_captions: Optional[Sequence[str]] = None,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    assert reward_model_prompt is None
    assert image_captions is None

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    validity = [True] * len(input_ids)

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for c, conversation, target in zip(
        range(len(conversations)), conversations, targets
    ):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        if mask_target:
            target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if mask_target:
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            final_query_len = cur_len
            final_response_len = round_len
            cur_len += round_len

        validity[c] = (
            validity[c]
            and (query_len is None or final_query_len <= query_len)
            and (response_len is None or final_response_len <= response_len)
        )
        if mask_target:
            target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                if mask_target:
                    target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        validity=validity,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    mask_target: bool = True,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    reward_model_prompt: Optional[str] = None,
    image_captions: Optional[Sequence[str]] = None,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

        reward_model_prompt_per_example = reward_model_prompt

        if (
            image_captions is not None
            and r"{factual_prompt}" in reward_model_prompt_per_example
        ):
            factual_prompt = FACTUAL_PROMPT
            for caption in image_captions[i]:
                factual_prompt = factual_prompt + f"  - {caption}\n"
            reward_model_prompt_per_example = reward_model_prompt_per_example.format(
                factual_prompt=factual_prompt
            )

        if reward_model_prompt_per_example is None:
            conversations.append(conv.get_prompt())
        else:
            conversations.append(
                conv.get_prompt() + reward_model_prompt_per_example + "</s>"
            )
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    validity = [True] * len(input_ids)

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for c, conversation, target in zip(
        range(len(conversations)), conversations, targets
    ):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        if mask_target:
            target[:cur_len] = IGNORE_INDEX

        final_query_len, final_response_len = 0, 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if mask_target:
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            final_query_len = cur_len
            final_response_len = round_len

            cur_len += round_len

        if final_response_len == 0:
            raise ValueError(f"Empty response: {conversation}")

        validity[c] = (
            validity[c]
            and (query_len is None or final_query_len <= query_len)
            and (response_len is None or final_response_len <= response_len)
        )

        if mask_target:
            target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                if mask_target:
                    target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    if reward_model_prompt is None:
        return dict(
            input_ids=input_ids,
            labels=targets,
            validity=validity,
        )
    else:
        return dict(
            input_ids=input_ids[:, :-1],
            labels=targets[:, :-1],
            validity=validity,
        )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    mask_target: bool = True,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    reward_model_prompt: Optional[str] = None,
    image_captions: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    # TODO(sheng): hack for LLAVA_V1, LLAMA_2
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(
            sources,
            tokenizer,
            has_image=has_image,
            mask_target=mask_target,
            query_len=query_len,
            response_len=response_len,
            reward_model_prompt=reward_model_prompt,
            image_captions=image_captions,
        )
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(
            sources,
            tokenizer,
            has_image=has_image,
            mask_target=mask_target,
            query_len=query_len,
            response_len=response_len,
            reward_model_prompt=reward_model_prompt,
            image_captions=image_captions,
        )

    raise NotImplementedError
