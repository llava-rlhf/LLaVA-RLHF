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

import abc
import copy
import dataclasses
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import einops
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import accelerate
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import convert_outputs_to_fp32

import transformers
from transformers.trainer_utils import enable_full_determinism, set_seed

from data_utils.data_utils_ppo import QueryResponseDataset
import data_utils.common_utils as utils
import models.distributed_utils as distributed_utils
from models.trainer_utils import create_optimizer, create_scheduler

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


logger = logging.getLogger(__name__)

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler

FIRST_STEP_IDX = 1


class KLController(abc.ABC):
    value: Union[int, float]

    def step(self, *args, **kwargs):
        pass


class FixedKLController(KLController):
    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = kl_coef


@dataclasses.dataclass
class HFDecodingArguments:
    """Only the core args for decoding with HF models."""

    top_p: float = 0.9
    top_k: int = 0
    temperature: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    max_new_tokens: int = (
        100  # This is aligned with `openai_utils.OpenAIDecodingArguments`.
    )
    num_return_sequences: int = 1


class AlpacaAccelerator(accelerate.Accelerator):
    """Thin wrapper for accelerate.Accelerator."""

    def __repr__(self):
        return (
            f"Accelerator(\n"
            f"  state={self.state}, \n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps:.6f}, \n"
            f"  split_batches={self.split_batches}, \n"
            f"  step_scheduler_with_optimizer={self.step_scheduler_with_optimizer},\n"
            f")"
        )

    def unwrap_optimizer(self, optimizer: accelerate.accelerator.AcceleratedOptimizer):
        return optimizer.optimizer


class RLTrainer(object):
    def __init__(
        self,
        args,
        train_dataset: QueryResponseDataset,
        eval_dataset: QueryResponseDataset,
        data_collator: Callable,
        tokenizer: transformers.PreTrainedTokenizer,
        policy: nn.Module,
        accelerator: AlpacaAccelerator,
        ref_policy: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(RLTrainer, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.kl_ctl = FixedKLController(kl_coef=args.kl_coef)
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)
        enable_full_determinism(
            self.args.seed
        ) if self.args.full_determinism else set_seed(self.args.seed)
        self.reward_model_prompt = None
        self.reward_model_prompt_untokenized = None

        if self.args.reward_prompt_file is not None:
            with open(self.args.reward_prompt_file, "r") as f:
                self.reward_model_prompt_untokenized = " " + f.read().strip()
            self.reward_model_prompt = self.tokenizer.encode(
                self.reward_model_prompt_untokenized,
                return_tensors="pt",
                add_special_tokens=False,
            )

        self.image_to_caption_mapping = None
        if self.args.image_to_caption_file is not None:
            with open(self.args.image_to_caption_file, "r") as f:
                self.image_to_caption_mapping = json.load(f)

    @abc.abstractmethod
    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        raise NotImplementedError

    @property
    def optimizable_params(self):
        return [
            p
            for p in self.policy.parameters()
            if p.requires_grad and p.grad is not None
        ]

    @torch.inference_mode()
    def _compute_grad_norm(self):
        grad_norm = torch.stack([p.grad.norm(2) for p in self.optimizable_params]).norm(
            2
        )
        return grad_norm

    @torch.inference_mode()
    def _compute_param_norm(self):
        param_norm = torch.stack([p.norm(2) for p in self.optimizable_params]).norm(2)
        return param_norm

    def step_with_rollouts(self, rollouts):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        assert isinstance(self.optimizer, AcceleratedOptimizer), (
            "`optimizer` must be pushed through `accelerator.prepare`. "
            "Otherwise the `accelerator.accumulate` context manager won't correctly disable `zero_grad` or `step`."
        )
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts=rollouts)
        stats_list = []
        for epoch_idx in range(self.args.noptepochs):
            for batch_idx, rollouts_batch in tqdm.tqdm(
                enumerate(rollouts_dataloader, 1),
                total=len(rollouts_dataloader),
                disable=not self.accelerator.is_main_process,
                desc="gradstep",
            ):
                gc.collect()
                torch.cuda.empty_cache()
                with self.accelerator.accumulate(self.policy):
                    stats_for_this_step = {}
                    with self.accelerator.no_sync(self.policy):
                        policy_loss, policy_stats = self.compute_policy_loss(
                            rollouts_batch
                        )
                        stats_for_this_step.update(policy_stats)
                        self.accelerator.backward(policy_loss)

                    value_loss, value_stats = self.compute_value_loss(rollouts_batch)
                    stats_for_this_step.update(value_stats)
                    self.accelerator.backward(value_loss)

                    if self.accelerator.sync_gradients:
                        # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                        if self.args.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.policy.parameters(), self.args.max_grad_norm
                            )
                        stats_for_this_step[
                            "loss/grad_norm"
                        ] = self._compute_grad_norm()
                        stats_list.append(stats_for_this_step)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        return utils.merge_dict(
            stats_list, torch.stack
        )  # list of dict -> dict: str -> 1-D tensor

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [
            next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)
        ]
        rollouts = self.rollout(queries_batches)
        train_stats = self.step_with_rollouts(rollouts)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = self.record_step_stats(
            rollouts=rollouts,
            train_stats=train_stats,
            step_idx=step_idx,
            kl_coef=self.kl_ctl.value,
        )
        self.kl_ctl.step(stats["objective/kl_sum_seq"])
        return stats

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = create_optimizer(
            args=self.args, model=self.policy, optimizer=self.optimizer
        )
        lr_scheduler = create_scheduler(
            args=self.args,
            optimizer=optimizer,
            lr_scheduler=self.lr_scheduler,
            num_training_steps=num_training_steps,
        )
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            optimizer, lr_scheduler
        )
        self.accelerator.register_for_checkpointing(
            self.lr_scheduler
        )  # LR scheduler needs another call to save.
        return self.optimizer, self.lr_scheduler

    def train(self, resume_training_ckpt: Optional[str] = None):
        """Entry point for training."""
        total_epochs = self.args.total_epochs
        total_episodes = len(self.train_dataset) * total_epochs  # noqa
        total_steps = total_episodes // self.args.rollout_batch_size  # noqa
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}"
        )

        self.create_optimizer_and_scheduler(total_steps)

        skipping_steps = 0
        if resume_training_ckpt is not None:
            skipping_steps = self.resume_training(resume_training_ckpt)
            print(
                f"Resuming training from {resume_training_ckpt} at step {skipping_steps}."
            )

        infinite_train_dataloader = self.get_train_dataloader()
        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, total_steps + FIRST_STEP_IDX),
            disable=not self.accelerator.is_main_process,
            desc="steps",
            total=total_steps,
        ):
            if step_idx < skipping_steps:
                for _ in range(self.args.rollout_accumulation_steps):
                    next(infinite_train_dataloader)
                continue

            if (
                step_idx % self.args.save_steps == 0
                or step_idx in self.args.save_steps_extra_list
            ):
                if step_idx > skipping_steps:
                    self.save_model(
                        os.path.join(self.args.output_dir, f"checkpoint-{step_idx}")
                    )
            if (
                self.args.eval_steps is not None
                and step_idx % self.args.eval_steps == 0
            ):
                self.evaluate(step_idx)
            self.log_history.append(self.step(infinite_train_dataloader, step_idx))
        return self.log_history

    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def resume_training(self, checkpoint_dir: str):
        raise NotImplementedError

    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        logger.warning(
            f"Batch size of {loader_name} dataloader: {batch_size}",
            # main_process_only=True,
        )

    def get_train_dataloader(self):
        logger.warning(
            f"Train dataset size: {len(self.train_dataset)}",
            # main_process_only=True
        )  # noqa
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  # noqa
        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    def get_rollouts_dataloader(
        self, rollouts: Dict[str, torch.Tensor], shuffle=True, drop_last=True, keys=None
    ):
        if keys is None:
            keys = tuple(rollouts.keys())

        def collate_rollouts(instances: Sequence[tuple]):
            return {
                key: torch.stack([instance[idx] for instance in instances])
                for idx, key in enumerate(keys)
            }

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])
        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=self.args.step_per_device_batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Do not prepare, since we don't need to shard the rollouts sampled on each batch.
        return rollouts_dataloader

    def get_policy_meta_prompts(self):
        if self.policy_meta_prompts is not None:
            return self.policy_meta_prompts

        policy_meta_prompt_pattern = self.args.data_config.policy_meta_prompt_pattern
        policy_meta_prompts = utils.make_meta_prompts(
            policy_meta_prompt_pattern,
        )
        self.policy_meta_prompts = policy_meta_prompts
        return policy_meta_prompts

    def get_reward_meta_prompts(self):
        if self.reward_meta_prompts is not None:
            return self.reward_meta_prompts

        reward_meta_prompt_pattern = self.args.data_config.reward_meta_prompt_pattern
        reward_meta_prompts = utils.make_meta_prompts(
            reward_meta_prompt_pattern,
        )
        self.reward_meta_prompts = reward_meta_prompts
        return reward_meta_prompts

    def get_principle_collection(self):
        if self.principle_collection is not None:
            return self.principle_collection

        principle_collection_path = self.args.data_config.principle_collection_path
        assert os.path.exists(principle_collection_path)
        print("Loading principle collection from", principle_collection_path)
        with open(principle_collection_path, "r") as f:
            principle_collection = json.load(f)
        self.principle_collection = principle_collection
        return principle_collection

    def prepare_reward_inputs(self, inputs, outputs):
        reward_meta_prompt = self.get_reward_meta_prompts()[0]

        if "{Dimensions}" in reward_meta_prompt:
            principle_collection = self.get_principle_collection()
            random.shuffle(principle_collection)
            dimension_str = []
            for item in principle_collection:
                dimension_str.append(f"- {item['definition']}")
            if self.args.data_config.max_principles is not None:
                if "weight" not in principle_collection[0]:
                    dimension_str = dimension_str[
                        : self.args.data_config.max_principles
                    ]
                else:
                    remaining_weights = [
                        item["weight"] for item in principle_collection
                    ]
                    remaining_idx = list(range(len(dimension_str)))

                    sampled_dimension_str = []
                    while (
                        len(sampled_dimension_str)
                        < self.args.data_config.max_principles
                    ):
                        sampled_idx = random.choices(
                            list(range(len(remaining_idx))), weights=remaining_weights
                        )[0]
                        sampled_dimension_str.append(
                            dimension_str[remaining_idx[sampled_idx]]
                        )
                        remaining_idx.pop(sampled_idx)
                        remaining_weights.pop(sampled_idx)

                    dimension_str = sampled_dimension_str

            dimension_str = "\n".join(dimension_str)
            reward_meta_prompt = reward_meta_prompt.replace(
                "{Dimensions}", dimension_str
            )

        return reward_meta_prompt.format(
            Input=inputs,
            Output=outputs,
        )


@dataclasses.dataclass
class NullCharCleanUp(object):
    def __call__(self, string: str):
        return string.replace("\x00", "")

    def __repr__(self):
        return "NullCharCleanUp cleans up the NULL chars to prevent db write failures due to encoding discrepancy."


def cast_with_native_amp(
    func: Callable, mixed_precision: Optional[str] = None
) -> Callable:
    """Almost like how huggingface accelerate cast `model.forward`."""
    if mixed_precision not in ("fp16", "bf16"):
        logger.warning(
            f"Unknown mixed precision mode: {mixed_precision}, falling back to fp32."
        )
        return func

    if mixed_precision == "fp16":
        output_func = torch.cuda.amp.autocast(dtype=torch.float16)(func)
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        output_func = torch.autocast(device_type=device_type, dtype=torch.bfloat16)(
            func
        )
    output_func = convert_outputs_to_fp32(output_func)
    return output_func


def remove_image_token(completions):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != IMAGE_TOKEN_INDEX]
        clean_completions[idx] = completion
    return clean_completions


def truncate_after_eos(completions, eos_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != IMAGE_TOKEN_INDEX]
        clean_completions[idx] = completion
        try:
            end_idx = completion.index(eos_token_id)
            clean_completions[idx] = completion[: end_idx + 1]
        except ValueError:
            pass
    return clean_completions


def truncate_after_eos_with_padding(
    completions, eos_token_id, pad_token_id, additional_tokens=None
):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        try:
            end_idx = completion.index(eos_token_id)
        except ValueError:
            end_idx = None

        if additional_tokens is not None:
            for additional_token in additional_tokens:
                try:
                    end_idx = completion.index(additional_token)
                except ValueError:
                    pass

        if end_idx is not None:
            clean_completions[idx] = completion[: end_idx + 1]

            if end_idx + 1 < len(completion):
                clean_completions[idx] = clean_completions[idx] + [pad_token_id] * (
                    len(completion) - end_idx - 1
                )

    clean_completions = torch.tensor(
        clean_completions, dtype=torch.long, device=completions.device
    )
    return clean_completions
