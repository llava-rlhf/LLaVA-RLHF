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


import json
import gc
import glob
from itertools import chain
import logging
import os
import pathlib
import random
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import accelerate
import pandas as pd
import torch
import tqdm
import transformers

from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from data_utils.data_utils_ppo import QueryResponseDataset

import data_utils.common_utils as common_utils

from data_utils.constants import AnswerType, FACTUAL_PROMPT

import models.rl_models as rl_models

from models.qlora_model import load_4bit_model_for_inference
from models.reward_model import load_4bit_reward_model_for_inference
from models.rl_trainer import (
    AlpacaAccelerator,
    RLTrainer,
    remove_image_token,
    truncate_after_eos_with_padding,
)


AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyPathOrNone = Optional[AnyPath]

logger = logging.getLogger(__name__)

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler


# Name of the files used for checkpointing
ADAPTER_MODEL_DIR = "adapter_model"
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
VALUE_HEAD_NAME = "value_head.pt"
SCALER_NAME = "scaler.pt"


class PPOTrainer(RLTrainer):
    def __init__(
        self,
        args,
        train_dataset: QueryResponseDataset,
        eval_dataset: QueryResponseDataset,
        data_collator: Callable,
        policy: rl_models.ActorCritic,
        ref_policy: rl_models.Policy,
        reward_model,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: AlpacaAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(PPOTrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def _shape_reward(
        self,
        rewards: torch.Tensor,
        responses: torch.Tensor,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        length_bonus: torch.Tensor,
        correct_bonus: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # For some reason, line below doesn't work.
        # kl = (logits.softmax(dim=-1) * (logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))).sum(dim=-1)

        if self.args.kl_approximator == "k1":
            # KL (q | p) = sum_i q_i (log q_i - log p_i)
            kl = torch.clamp(logprobs - ref_logprobs, min=0.0)
        elif self.args.kl_approximator == "k3":
            # r = p / q, log r = log p - log q
            # KL (q | p) = (r - 1) - log r = e ^ log r - 1 - log r
            log_r = ref_logprobs - logprobs
            kl = torch.exp(log_r) - 1.0 - log_r
        else:
            raise ValueError(f"Unknown KL approximator: {self.args.kl_approximator}")

        non_score_rewards = -self.kl_ctl.value * kl
        shaped_rewards = non_score_rewards.clone()
        # This introduces a small index off by one bug if pad_token_id == eos_token_id.
        # terminal_positions = (responses != self.tokenizer.pad_token_id).sum(dim=1) - 1
        # shaped_rewards[list(range(rewards.size(0))), terminal_positions] += rewards

        shaped_rewards[:, -1] += (
            rewards
            + (length_bonus * self.args.length_bonus_score)
            + (correct_bonus * self.args.correct_bonus_score)
            + self.args.reward_bias
        )
        return dict(
            shaped_rewards=shaped_rewards, non_score_rewards=non_score_rewards, kl=kl
        )

    def _estimate_advantage(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        if self.args.whiten_rewards:
            rewards = whiten(
                rewards, shift_mean=False, async_stats=self.args.whitening_async_stats
            )
        else:
            rewards = rewards * 10.0
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.response_len
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = whiten(
            advantages, shift_mean=True, async_stats=self.args.whitening_async_stats
        )
        return dict(returns=returns, advantages=advantages)

    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        # Give up dropout throughout.
        self.policy.eval()
        # `keep_fp32_wrapper` retains the autocast wrapper of model.forward created by accelerate:
        #  recall one sets mixed precision options with accelerator.
        # The precise value of this arg doesn't matter here, since we use the unwrapped model only for respond.
        # Generally, try to use the wrapped model as much as you can, since it's got the autocast/cast-back wrappers.
        unwrapped_policy = self.accelerator.unwrap_model(
            self.policy, keep_fp32_wrapper=True
        )

        self.ref_policy.eval()
        self.reward_model.eval()

        rollouts = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            total=len(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            gc.collect()
            torch.cuda.empty_cache()
            # Sample rollouts.
            (
                images,
                reward_images,
                image_file_ids,
                caption_types,
                length_bonus_multiplier,
                queries,
                query_attn_masks,
            ) = common_utils.unpack_dict(
                common_utils.prepare_inputs(batch, device=self.accelerator.device),
                keys=(
                    "images",
                    "reward_images",
                    "image_file_ids",
                    "caption_types",
                    "length_bonus_multiplier",
                    "queries",
                    "query_attn_masks",
                ),
            )

            if self.args.bf16:
                images = images.to(torch.bfloat16)
                reward_images = reward_images.to(torch.bfloat16)
            elif self.args.fp16:
                images = images.half()
                reward_images = reward_images.half()

            respond_outputs = unwrapped_policy.respond(
                queries, query_attn_masks, images, temperature=self.args.temperature
            )
            (responses,) = common_utils.unpack_dict(respond_outputs, ("responses",))

            additional_token1 = self.tokenizer.encode("?", add_special_tokens=False)[0]
            assert additional_token1 == 1577

            additional_token2 = self.tokenizer.encode("\n?")[-1]
            assert additional_token2 == 29973

            responses = truncate_after_eos_with_padding(
                responses,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                additional_tokens=[additional_token1, additional_token2],
            )

            rollouts_batch = {
                "images": images,
                "reward_images": reward_images,
                "queries": queries,
                "query_attn_masks": query_attn_masks,
                "responses": responses,
            }

            text_responses = self.tokenizer.batch_decode(
                responses,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            correct_bonus = []
            for idx, response in enumerate(text_responses):
                caption_type = AnswerType(caption_types[idx].item())

                if caption_type == AnswerType.GENERAL:
                    correct_bonus.append(0.0)
                elif caption_type in [AnswerType.A_IN_ABCD, AnswerType.B_IN_ABCD, AnswerType.C_IN_ABCD, AnswerType.D_IN_ABCD]:
                    expected_start = caption_type.name.split("_")[0] + "."
                    expected_phrase = "correct option is " + expected_start
                    if response.strip().startswith(expected_start) or expected_phrase in response:
                        correct_bonus.append(1.0)
                    else:
                        correct_bonus.append(0.0)
                elif caption_type == AnswerType.NO_IN_YESNO:
                    if response.strip().startswith("No"):
                        correct_bonus.append(0.5)
                    elif response.strip().startswith("Yes"):
                        correct_bonus.append(-0.5)
                    else:
                        correct_bonus.append(0.0)
                elif caption_type == AnswerType.YES_IN_YESNO:
                    # TODO(zhiqings): for now, we do not give symbolic award for "Yes" in Yes/No questions.
                    correct_bonus.append(0.0)
                else:
                    raise NotImplementedError
            assert len(correct_bonus) == len(text_responses)
            correct_bonus = torch.tensor(correct_bonus, device=responses.device)

            has_stop_token = [
                self.tokenizer.eos_token_id in response
                for response in responses.tolist()
            ]

            sequences = [
                torch.concat((query, response), dim=0)
                for query, response in zip(queries, responses)
            ]
            sequences = torch.stack(sequences, dim=0)

            sequences = remove_pad_and_left_pad(
                sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if self.reward_model_prompt is not None:
                if self.image_to_caption_mapping is None:
                    reward_model_prompt = (
                        self.reward_model_prompt.reshape(1, -1)
                        .repeat(len(sequences), 1)
                        .to(self.accelerator.device)
                    )
                    sequences = torch.cat((sequences, reward_model_prompt), dim=1)
                else:
                    reward_model_prompt_untokenized = (
                        self.reward_model_prompt_untokenized
                    )
                    image_to_caption_mapping = self.image_to_caption_mapping

                    image_ids = []
                    for i in range(len(sequences)):
                        image_file = str(image_file_ids[i].item()).zfill(12) + ".jpg"
                        caption_type = AnswerType(caption_types[i].item())
                        if caption_type in [AnswerType.GENERAL, AnswerType.NO_IN_YESNO, AnswerType.YES_IN_YESNO]:
                            image_id = image_file
                        elif caption_type in [AnswerType.A_IN_ABCD, AnswerType.B_IN_ABCD, AnswerType.C_IN_ABCD, AnswerType.D_IN_ABCD]:
                            image_id = "aok_" + image_file
                        else:
                            print(caption_type)
                            print([AnswerType.GENERAL, AnswerType.NO_IN_YESNO, AnswerType.YES_IN_YESNO])
                            print([AnswerType.A_IN_ABCD, AnswerType.B_IN_ABCD, AnswerType.C_IN_ABCD, AnswerType.D_IN_ABCD])
                            raise NotImplementedError
                        image_ids.append(image_id)

                    captions = [
                        image_to_caption_mapping[image_id] for image_id in image_ids
                    ]

                    assert r"{factual_prompt}" in reward_model_prompt_untokenized

                    reward_model_prompts = []

                    for caption_list in captions:
                        caption_list = caption_list[:]
                        random.shuffle(caption_list)
                        factual_prompt = FACTUAL_PROMPT
                        for caption in caption_list:
                            factual_prompt = factual_prompt + f"  - {caption}\n"
                        reward_model_prompt_per_example = (
                            reward_model_prompt_untokenized.format(
                                factual_prompt=factual_prompt
                            )
                        )
                        reward_model_prompts.append(reward_model_prompt_per_example)
                    reward_model_prompts = self.tokenizer(
                        reward_model_prompts,
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding="longest",
                    )["input_ids"]
                    reward_model_prompts = reward_model_prompts.to(
                        self.accelerator.device
                    )

                    sequences = torch.cat((sequences, reward_model_prompts), dim=1)
                    sequences = remove_pad_and_left_pad(
                        sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            clean_sequences = sequences.detach().clone()
            clean_sequences[
                clean_sequences == IMAGE_TOKEN_INDEX
            ] = self.tokenizer.eos_token_id

            text_sequences = self.tokenizer.batch_decode(
                clean_sequences,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if self.accelerator.is_main_process:
                print("=" * 20)
                print(text_sequences[0].split("<unk><s> ")[-1])
                print("=" * 20)
                image_id = image_file_ids[0].item()
                # convert int into "000000xxxxxx.jpg"
                image_id = (
                    "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/"
                    + str(image_id).zfill(12)
                    + ".jpg"
                )
                print(image_id)
                print("=" * 20)

            non_pad_mask = responses.ne(self.tokenizer.pad_token_id)
            non_pad_seq_len = (
                non_pad_mask.sum(dim=1).float().to(self.accelerator.device)
            )
            length_bonus = non_pad_seq_len / float(self.args.response_len)

            # convert length_bonus_multiplier to the shape, type, and device of length_bonus
            length_bonus = length_bonus * length_bonus_multiplier.to(
                length_bonus.device
            ).reshape(length_bonus.shape).to(length_bonus.dtype)

            sequences_attention_mask = sequences.ne(self.tokenizer.pad_token_id)

            # Evaluate logprobs of the samples.
            batch_size_per_device = rollouts_batch["responses"].shape[0]
            sub_batch_size = self.args.reward_model_per_device_batch_size
            if sub_batch_size is None or sub_batch_size == batch_size_per_device:
                policy_outputs = self.policy(
                    **rollouts_batch, temperature=self.args.temperature
                )
            else:
                assert batch_size_per_device % sub_batch_size == 0

                policy_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    sub_batch = {
                        key: value[
                            sub_batch_idx
                            * sub_batch_size : (sub_batch_idx + 1)
                            * sub_batch_size
                        ]
                        for key, value in rollouts_batch.items()
                    }
                    sub_batch_policy_outputs = self.policy(
                        **sub_batch, temperature=self.args.temperature
                    )
                    policy_outputs_list.append(sub_batch_policy_outputs)

                policy_outputs = common_utils.merge_dict(
                    policy_outputs_list, merge_fn=torch.cat
                )
                del sub_batch_policy_outputs
                del policy_outputs_list
                del sub_batch

            if sub_batch_size is None or sub_batch_size == batch_size_per_device:
                ref_policy_outputs = self.ref_policy(
                    **rollouts_batch, temperature=self.args.temperature
                )
            else:
                assert batch_size_per_device % sub_batch_size == 0

                ref_policy_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    sub_batch = {
                        key: value[
                            sub_batch_idx
                            * sub_batch_size : (sub_batch_idx + 1)
                            * sub_batch_size
                        ]
                        for key, value in rollouts_batch.items()
                    }
                    sub_batch_ref_policy_outputs = self.ref_policy(
                        **sub_batch, temperature=self.args.temperature
                    )
                    ref_policy_outputs_list.append(sub_batch_ref_policy_outputs)

                ref_policy_outputs = common_utils.merge_dict(
                    ref_policy_outputs_list, merge_fn=torch.cat
                )
                del sub_batch_ref_policy_outputs
                del ref_policy_outputs_list
                del sub_batch

            policy_outputs = common_utils.unpack_dict(
                policy_outputs,
                keys=("logprobs", "values", "entropies"),
                return_type=dict,
            )

            ref_policy_outputs = common_utils.unpack_dict(
                ref_policy_outputs, keys=("logprobs", "entropies"), return_type=dict
            )
            rollouts_batch.update(policy_outputs)
            rollouts_batch.update(
                {f"ref_{key}": value for key, value in ref_policy_outputs.items()}
            )

            rollouts_batch["length_bonus"] = length_bonus
            rollouts_batch["correct_bonus"] = correct_bonus

            if sub_batch_size is None or sub_batch_size == batch_size_per_device:
                reward_outputs = self.reward_model(
                    input_ids=sequences,
                    attention_mask=sequences_attention_mask,
                    images=reward_images,
                )
            else:
                assert batch_size_per_device % sub_batch_size == 0

                reward_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    idx_start = sub_batch_idx * sub_batch_size
                    idx_end = (sub_batch_idx + 1) * sub_batch_size
                    sub_batch_reward_outputs = self.reward_model(
                        input_ids=sequences[idx_start:idx_end],
                        attention_mask=sequences_attention_mask[idx_start:idx_end],
                        images=reward_images[idx_start:idx_end],
                    )
                    reward_outputs_list.append(sub_batch_reward_outputs)

                reward_outputs = common_utils.merge_dict(
                    reward_outputs_list, merge_fn=torch.cat
                )
                del reward_outputs_list
                del sub_batch_reward_outputs

            reward_outputs = self.post_reward(
                reward_outputs,
                responses,
                penalize_no_stop_token=self.args.penalize_no_stop_token,
                relative_stop_token_penalty=self.args.relative_stop_token_penalty,
                has_stop_token=has_stop_token,
            )

            rollouts_batch.update(reward_outputs)

            # Shape reward with KL penalty.
            shape_reward_outputs = self._shape_reward(
                rewards=rollouts_batch["rewards"],
                responses=rollouts_batch["responses"],
                logprobs=rollouts_batch["logprobs"],
                ref_logprobs=rollouts_batch["ref_logprobs"],
                length_bonus=rollouts_batch["length_bonus"],
                correct_bonus=rollouts_batch["correct_bonus"],
            )
            rollouts_batch.update(shape_reward_outputs)

            rollouts_batch_cpu = {
                key: value.cpu() for key, value in rollouts_batch.items()
            }
            rollouts.append(rollouts_batch_cpu)

        # Items in dict need to be of same shape.
        rollouts = common_utils.merge_dict(rollouts, merge_fn=torch.cat)

        # Estimating advantages outside the loop gives more samples for reward normalization.
        advantages = self._estimate_advantage(
            rewards=rollouts["shaped_rewards"].to(self.accelerator.device),
            values=rollouts["values"].to(self.accelerator.device),
        )
        advantages = {key: value.cpu() for key, value in advantages.items()}
        return {**rollouts, **advantages}

    def post_reward(
        self,
        reward_outputs: Dict[str, torch.Tensor],
        responses: torch.Tensor,
        penalize_no_stop_token: bool,
        relative_stop_token_penalty: bool,
        has_stop_token: List[bool],
    ) -> Dict[str, torch.Tensor]:
        """Assign bad reward values to sequences which didn't stop properly."""
        if penalize_no_stop_token:
            has_stop_token = torch.tensor(has_stop_token, device=responses.device)
            rewards = reward_outputs["rewards"]
            if relative_stop_token_penalty:
                rewards = (
                    rewards + (~has_stop_token).float() * self.args.penalty_reward_value
                )
            else:
                rewards[~has_stop_token] = self.args.penalty_reward_value
            reward_outputs["rewards"] = rewards
            return reward_outputs

        if self.args.truncate_token_ids is None:
            return reward_outputs

        def get_validity_mask(
            sequences: torch.Tensor, end_token_id: int
        ) -> torch.Tensor:
            """Mark a batch element as False if the sequence doesn't end with `end_token_id` after `truncate_after`."""
            assert sequences.dim() == 2
            validity_mask = []
            for sequence in sequences:
                (nonzeros,) = (sequence == end_token_id).nonzero(as_tuple=True)
                if len(nonzeros) == 0:
                    validity_mask.append(False)
                else:
                    validity_mask.append(
                        self.args.truncate_after is None
                        or
                        # Last occurrence of `end_token_id` is after `truncate_after`.
                        nonzeros[-1] > self.args.truncate_after
                    )
            return torch.tensor(validity_mask, device=sequences.device)

        validity_masks = [
            get_validity_mask(responses, end_token_id)
            for end_token_id in self.args.truncate_token_ids
        ]
        validity_mask = torch.stack(validity_masks).any(
            dim=0
        )  # Sequence is valid if it ends with any end token.
        rewards = reward_outputs["rewards"]
        rewards[~validity_mask] = self.args.penalty_reward_value
        return reward_outputs

    def compute_policy_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            values,
            old_logprob,
            returns,
            advantages,
            queries,
            query_attn_masks,
            responses,
            images,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "values",
                    "logprobs",
                    "returns",
                    "advantages",
                    "queries",
                    "query_attn_masks",
                    "responses",
                    "images",
                ),
            ),
            device=self.accelerator.device,
        )

        # Enable training mode for graident checkpointing.
        self.policy.train()

        outputs = self.policy(
            queries,
            query_attn_masks,
            responses,
            images,
            temperature=self.args.temperature,
            mode="policy",
        )

        logprob = outputs["logprobs"]
        ratio = torch.exp(logprob - old_logprob)
        # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange
        )
        pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (
            (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()
        )  # noqa

        loss = pg_loss + outputs["dummy_loss"]

        entropy = outputs["entropies"].mean()
        approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

        return_mean, return_var = returns.mean(), returns.var(unbiased=False)

        stats = dict(
            loss=dict(policy=pg_loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
        )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    def compute_value_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            values,
            old_logprob,
            returns,
            advantages,
            queries,
            query_attn_masks,
            responses,
            images,
            reward_images,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "values",
                    "logprobs",
                    "returns",
                    "advantages",
                    "queries",
                    "query_attn_masks",
                    "responses",
                    "images",
                    "reward_images",
                ),
            ),
            device=self.accelerator.device,
        )

        # Enable training mode for graident checkpointing.
        self.policy.train()

        outputs = self.policy(
            queries,
            query_attn_masks,
            responses,
            images,
            reward_images,
            temperature=self.args.temperature,
            mode="value",
        )

        vpred = outputs["values"]
        vpredclipped = torch.clamp(
            vpred,
            min=values - self.args.cliprange_value,
            max=values + self.args.cliprange_value,
        )
        vf_losses1 = (vpred - returns) ** 2.0
        vf_losses2 = (vpredclipped - returns) ** 2.0
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()

        loss = self.args.vf_coef * vf_loss + outputs["dummy_loss"]

        value_mean, value_var = values.mean(), values.var(unbiased=False)

        stats = dict(
            loss=dict(value=vf_loss),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        kl = rollouts["kl"]
        kl_sum_seq, kl_avg_seq = kl.sum(dim=1).mean(dim=0), kl.mean()
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        stats = {
            f"objective/kl_coef": kwargs["kl_coef"],
            f"objective/kl_sum_seq": kl_sum_seq,
            f"objective/kl_avg_seq": kl_avg_seq,
            f"objective/length_bonus": rollouts["length_bonus"].mean(),
            f"objective/correct_bonus": rollouts["correct_bonus"].mean(),
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
            f"objective/lr": self.optimizer.param_groups[0]["lr"],
            f"objective/entropies": rollouts["entropies"].mean(),
            f"objective/ref_entropies": rollouts["ref_entropies"].mean(),
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
        stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in stats.items()
        }
        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)
            if self.args.output_dir is not None:
                # Store rollout data to disk to debug.
                rollouts_to_disk = {
                    key: self.tokenizer.batch_decode(
                        remove_image_token(
                            tensor,
                        ),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for key, tensor in common_utils.unpack_dict(
                        rollouts, keys=("queries", "responses"), return_type=dict
                    ).items()
                }

                rewards = [str(_) for _ in rollouts["rewards"].tolist()]
                rollouts_to_disk["rewards"] = rewards

                rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(
                    orient="records"
                )
                rollout_log_dir = os.path.join(self.args.output_dir, "rollouts")
                os.makedirs(rollout_log_dir, exist_ok=True)
                with open(
                    os.path.join(rollout_log_dir, f"step_{step_idx}.json"),
                    "w",
                ) as f:
                    json.dump(rollouts_to_disk, f, indent=4)
        return stats

    @torch.inference_mode()
    def save_model(
        self,
        output_dir: Optional[str] = None,
        give_rw_access=True,
        check_corrupted=True,
    ):
        output_dir = self.args.output_dir if output_dir is None else output_dir

        global_rank = int(os.environ.get("RANK", 0))

        if global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            print("Saving model checkpoint to %s" % output_dir)

            # Save policy model.
            unwrapped_policy = self.accelerator.unwrap_model(
                self.policy, keep_fp32_wrapper=True
            )

            policy_model = unwrapped_policy.policy.base_model

            peft_model_path = os.path.join(output_dir, ADAPTER_MODEL_DIR)

            # policy_model.save_pretrained(peft_model_path)
            save_adapters(
                policy_model,
                peft_model_path,
                adapter_names=["lora_policy"],
            )

            pytorch_model_paths = glob.glob(
                os.path.join(output_dir, "pytorch_model*.bin")
            )
            for pytorch_model_path in pytorch_model_paths:
                if os.path.exists(pytorch_model_path):
                    os.remove(pytorch_model_path)

            # Save value model.
            value_model = unwrapped_policy.value_model

            save_adapters(
                value_model.base_model,
                peft_model_path,
                adapter_names=["lora_value"],
            )

            torch.save(
                value_model.value_head.state_dict(),
                os.path.join(output_dir, VALUE_HEAD_NAME),
            )

            pytorch_model_paths = glob.glob(
                os.path.join(output_dir, "pytorch_model*.bin")
            )
            for pytorch_model_path in pytorch_model_paths:
                if os.path.exists(pytorch_model_path):
                    os.remove(pytorch_model_path)

            # Save optimizer.
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            # Save scheduler.
            torch.save(
                self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME)
            )

            # Delete other optimizer checkpoints to save disk space.
            # glob pattern to match all optimizer.pt files in the father directory
            pattern = os.path.join(os.path.dirname(output_dir), "*/optimizer.pt")

            # get a list of all matching paths
            optimizer_files = glob.glob(pattern)

            # iterate over the optimizer files
            for file in optimizer_files:
                # if the file is not in the output_dir, delete it
                if output_dir not in file:
                    os.remove(file)

        else:
            print("Skipping PEFT checkpoint save on rank %d" % global_rank)

    @torch.inference_mode()
    def resume_training(self, checkpoint_dir):
        # Load optimizer.
        optimizer_path = os.path.join(checkpoint_dir, OPTIMIZER_NAME)
        if os.path.exists(optimizer_path):
            load_paged_optimizer_state_dict(
                self.optimizer.optimizer,
                torch.load(
                    optimizer_path,
                    map_location="cpu",
                ),
            )

        # Unpage optimizer state.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Load scheduler.
        scheduler_path = os.path.join(checkpoint_dir, SCHEDULER_NAME)
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(
                torch.load(
                    scheduler_path,
                    map_location="cpu",
                )
            )

        spattern = re.compile(r"checkpoint-(\d+)")
        skipping_steps = int(spattern.search(checkpoint_dir).group(1))
        return skipping_steps


def smart_tokenizer_and_embedding_resize(
    num_new_tokens: int,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)


def make_models(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    accelerator: accelerate.Accelerator,
    num_new_tokens: int = 0,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    def make_generative_policy(
        adapter_name, is_trainable, reuse_base_model=True, resume_path=None
    ):
        model = load_4bit_model_for_inference(
            checkpoint_dir=resume_path or args.policy_model_name_or_path,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=args.image_grid_pinpoints,
            bits=4,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            reuse_base_model=reuse_base_model,
            trust_remote_code=args.trust_remote_code,
        )
        smart_tokenizer_and_embedding_resize(num_new_tokens, tokenizer, model)
        return model

    def make_reward_model(
        adapter_name, is_trainable, reuse_base_model=True, resume_path=None
    ):
        model = load_4bit_reward_model_for_inference(
            checkpoint_dir=resume_path or args.reward_model_name_or_path,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=args.image_grid_pinpoints,
            bits=4,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            reuse_base_model=reuse_base_model,
            trust_remote_code=args.trust_remote_code,
        )
        smart_tokenizer_and_embedding_resize(
            num_new_tokens, tokenizer, model.backbone_model
        )
        return model

    policy_resume_path = None
    if resume_from_checkpoint:
        policy_resume_path = os.path.join(
            resume_from_checkpoint, ADAPTER_MODEL_DIR, "lora_policy"
        )

    policy = rl_models.make_policy_with_base_model(
        args,
        make_generative_policy(
            adapter_name="lora_policy",
            is_trainable=True,
            resume_path=policy_resume_path,
        ),
        tokenizer,
        adapter_name="lora_policy",
    )

    value_resume_path = None
    value_head_resume_path = None
    if resume_from_checkpoint:
        value_resume_path = os.path.join(
            resume_from_checkpoint, ADAPTER_MODEL_DIR, "lora_value"
        )
        value_head_resume_path = os.path.join(resume_from_checkpoint, VALUE_HEAD_NAME)

    if args.init_value_with_reward:
        # Initialize value from reward model a la OAI.
        logger.warning("Initializing value model with reward model.")
        value_model = rl_models.make_value_with_base_model(
            args,
            make_reward_model(
                adapter_name="lora_value",
                is_trainable=True,
                # reuse_base_model=False,
                resume_path=value_resume_path,
            ).backbone_model,
            tokenizer,
            adapter_name="lora_value",
        )
    else:
        logger.warning("Initializing value model with policy model.")
        # Initialize value from policy. Works for sanity, but generally performs worse in instruction-following.
        value_model = rl_models.make_value_with_base_model(
            args,
            make_generative_policy(
                adapter_name="lora_value",
                is_trainable=True,
                # reuse_base_model=False,
                resume_path=value_resume_path,
            ),
            tokenizer,
            adapter_name="lora_value",
        )

    if value_head_resume_path:
        value_model.value_head.load_state_dict(
            torch.load(value_head_resume_path, map_location="cpu")
        )

    actor_critic = rl_models.ActorCritic(policy=policy, value_model=value_model)
    # We cast how respond should run. It's important the dtypes be consistent with training, since a bf16
    # fine-tuned model might not work with fp16 inference.
    # Cast step below must precede accelerator.prepare(), since wrapped model might not have `respond` method.
    # actor_critic = common.prepare_model_for_custom_fn(
    #     model=actor_critic, fn_name="respond", accelerator=accelerator
    # )

    ref_policy = rl_models.make_policy_with_base_model(
        args,
        make_generative_policy(
            adapter_name="lora_ref_policy",
            is_trainable=False,
        ),
        tokenizer,
        adapter_name="lora_ref_policy",
    )

    reward_model = make_reward_model(
        adapter_name="lora_reward",
        is_trainable=False,
    )

    if args.vision_tower is not None:
        reward_model.backbone_model.config.image_aspect_ratio = args.image_aspect_ratio
        reward_model.backbone_model.config.image_grid_pinpoints = (
            args.image_grid_pinpoints
        )

        vision_tower = reward_model.backbone_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
        vision_tower.requires_grad_(False)

        mm_projector = reward_model.backbone_model.get_model().mm_projector
        mm_projector.to(device="cuda", dtype=torch.bfloat16)
        mm_projector.requires_grad_(False)

        if args.vision_tower == "different":
            policy_vision_tower = policy.base_model.get_vision_tower()
            if not policy_vision_tower.is_loaded:
                policy_vision_tower.load_model()
            policy_vision_tower.to(device="cuda", dtype=torch.bfloat16)
            policy_vision_tower.requires_grad_(False)

            policy_mm_projector = policy.base_model.get_model().mm_projector
            policy_mm_projector.to(device="cuda", dtype=torch.bfloat16)
            policy_mm_projector.requires_grad_(False)

    actor_critic = accelerator.prepare(actor_critic)  # noqa
    if not args.init_value_with_reward:
        reward_model = accelerator.prepare(reward_model)

    # # TODO: This is a hack to get FSDP running. Remove in the future when we figure things out.
    # if accelerator.distributed_type == accelerate.DistributedType.FSDP:
    #     inputs = tokenizer("fsdp are you happy now??? :)" * 50, return_tensors="pt")
    #     inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    #     actor_critic(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])
    return dict(policy=actor_critic, ref_policy=ref_policy, reward_model=reward_model)


def whiten(
    values: torch.Tensor, shift_mean=True, epsilon=1e-8, async_stats="full_batch"
) -> torch.Tensor:
    assert async_stats in ["full_batch", "per_gpu", "none"]

    values_for_statistics = values
    if async_stats == "full_batch":
        if not values_for_statistics.is_cuda:
            raise ValueError("SyncWhiten expected input tensor to be on GPU")

        need_sync = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        if need_sync:
            process_group = torch.distributed.group.WORLD
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        if need_sync:
            tensor_list = [
                torch.zeros_like(values_for_statistics) for _ in range(world_size)
            ]
            torch.distributed.all_gather(tensor_list, values_for_statistics)
            values_for_statistics = torch.cat(tensor_list, dim=0)

    if async_stats in ["full_batch", "per_gpu"]:
        assert (
            values_for_statistics.size(0) >= 8
        ), f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
        mean = values_for_statistics.mean()  # noqa
        std = values_for_statistics.std(unbiased=False)  # noqa

    else:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, unbiased=False, keepdim=True)

    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def save_adapters(model, save_directory, adapter_names, **kwargs):
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        model: The model to save.
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        adapter_name (`str`):
            Name of the adapter to save.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
    os.makedirs(save_directory, exist_ok=True)
    # model.create_or_update_model_card(save_directory)

    for adapter_name, peft_config in model.peft_config.items():
        if adapter_name in adapter_names:
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                model,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    model.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode


def load_paged_optimizer_state_dict(optimizer, state_dict):
    """
    Load an optimizer state dict that was saved.
    """

    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict["param_groups"]

    if len(groups) != len(saved_groups):
        raise ValueError(
            "loaded state dict has a different number of " "parameter groups"
        )
    param_lens = (len(g["params"]) for g in groups)
    saved_lens = (len(g["params"]) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group "
            "that doesn't match the size of optimizer's group"
        )

    # Update the state
    id_map = {
        p: old_id
        for old_id, p in zip(
            chain.from_iterable(g["params"] for g in saved_groups),
            chain.from_iterable(g["params"] for g in groups),
        )
    }

    for g in groups:
        for p in g["params"]:
            if p in optimizer.state:
                values = optimizer.state[p]
                for k, v in values.items():
                    if isinstance(v, torch.Tensor):
                        v.copy_(state_dict["state"][id_map[p]][k])
                        optimizer.state[p][k] = v.to("cpu")
                    else:
                        optimizer.state[p][k] = state_dict["state"][id_map[p]][k]


def remove_pad_and_left_pad(completions, pad_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    padded_length = len(clean_completions[0])
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != pad_token_id]

        if len(completion) < padded_length:
            completion = [pad_token_id] * (padded_length - len(completion)) + completion

        clean_completions[idx] = completion

    clean_completions = torch.tensor(
        clean_completions, dtype=torch.long, device=completions.device
    )
    return clean_completions
