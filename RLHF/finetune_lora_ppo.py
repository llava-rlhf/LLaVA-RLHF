# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Optional, List
import logging

import torch
import torch.distributed as dist

import accelerate
from accelerate import DistributedDataParallelKwargs

import transformers
from transformers import set_seed

try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer

    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer

    print("Using slow tokenizer")

from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils.data_utils_ppo import make_rl_data_module
from lora_utils import (
    get_last_checkpoint,
    DEFAULT_PAD_TOKEN,
)
from models.ppo_trainer import (
    PPOTrainer,
    make_models,
)
from models.rl_trainer import AlpacaAccelerator


from llava import conversation as conversation_lib
from llava.model import *
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    reward_model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    policy_model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    base_model_name: Optional[str] = field(default="EleutherAI/pythia-12b")
    reward_base_model_name: Optional[str] = field(default=None)
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    freeze_mm_mlp_adapter: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default="alpaca_instructions")
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to stop generation with."},
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From AlpacaFarm
    truncate_tokens: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Tokens in strings to truncate at first occurrence. "
            "This was used in original OAI summarization paper to avoid models returning incomplete sentences. "
        },
    )
    truncate_after: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate after this number of tokens. Prevents early truncation."
        },
    )
    penalty_reward_value: float = field(
        default=-1.0,
        metadata={
            "help": "Reward assigned to sequences that are truncated, "
            "e.g., due to outputting incomplete sentences for given context window."
        },
    )
    penalize_no_stop_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that do not contain stop_token."
        },
    )
    length_bonus_score: float = field(
        default=0.0,
        metadata={
            "help": "Add the reward for longer sequences by this amount. "
            "This is useful for encouraging longer sequences."
        },
    )
    correct_bonus_score: float = field(
        default=0.0,
        metadata={
            "help": "Add the reward for denying the user request by this amount. "
        },
    )
    reward_bias: float = field(
        default=0.0,
        metadata={"help": "Add this amount to the reward."},
    )
    diverse_penalty_reward_scale: float = field(
        default=0.0,
        metadata={"help": "Scale the reward for diverse sequences by this amount."},
    )
    penalize_non_diverse_responses: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that are not diverse within the batch."
        },
    )
    relative_stop_token_penalty: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that do not contain stop_token "
            "with a relative penalty based on the original reward."
        },
    )
    clean_tokens_after_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to clean up tokens after the first occurrence of stop_token."
        },
    )
    suppress_eos_at_generation: bool = field(
        default=False,
        metadata={
            "help": "Whether to suppress the end-of-sequence token at generation time."
        },
    )
    total_epochs: int = field(default=10)
    rollout_batch_size: int = field(default=512)
    step_batch_size: int = field(default=256)
    rollout_per_device_batch_size: int = field(default=32)
    step_per_device_batch_size: int = field(default=2)
    reward_model_per_device_batch_size: int = field(default=None)
    noptepochs: int = field(default=2)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    whiten_rewards: bool = field(default=True)
    temperature: float = field(default=1.0)
    kl_coef: float = field(default=0.2)
    kl_approximator: str = field(default="k1")
    target_kl: float = field(default=6.0)
    k_beta: float = field(default=0.1)
    adaptive_kl: bool = field(default=False)
    eval_batches: int = field(
        default=sys.maxsize,
        metadata={"help": "Maximum number of batches to evaluate on."},
    )
    init_value_with_reward: bool = field(
        default=True,
        metadata={"help": "Initialize the value model with the reward model."},
    )
    save_steps_extra: Optional[str] = field(
        default=None,
        metadata={
            "help": "A list of predetermined checkpoints to save, represented in the format 'no1__no2__no3'. "
            "Parse this with str.split('__')."
        },
    )
    query_len: int = field(default=128)
    min_token_limit: int = field(default=None)
    response_len: int = field(default=384)
    model_max_length: int = field(default=1024)
    whitening_async_stats: str = field(
        default="per_gpu",
        metadata={"help": "How to sync statistics for reward whitening."},
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: str = field(
        default=None,
        metadata={
            "help": "The directory to resume from. If None, will start from scratch."
        },
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Checks on rollout_batch_size only matter for PPO.
        assert (
            self.rollout_batch_size >= self.rollout_per_device_batch_size * world_size
        ), (
            "rollout_batch_size is smaller than rollout_per_device_batch_size * world_size. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.rollout_batch_size % (self.rollout_per_device_batch_size * world_size)
            == 0
        ), "rollout_batch_size is not a multiple of rollout_per_device_batch_size * world_size. "

        assert self.step_batch_size >= self.step_per_device_batch_size * world_size, (
            "step_batch_size is smaller than step_per_device_batch_size * world_size. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.step_batch_size % (self.step_per_device_batch_size * world_size) == 0
        ), "step_batch_size is not a multiple of step_per_device_batch_size * world_size. "

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.warning(
                f"Rollout stats:\n"
                f"\trollout_batch_size: {self.rollout_batch_size}\n"
                f"\trollout_per_device_batch_size: {self.rollout_per_device_batch_size}\n"
                f"\tworld_size: {world_size}\n",
            )
        assert (
            self.rollout_batch_size // self.rollout_per_device_batch_size
        ) % world_size == 0
        self.rollout_accumulation_steps = (
            self.rollout_batch_size // self.rollout_per_device_batch_size // world_size
        )

        logger.warning(
            f"Step stats:\n"
            f"\tstep_batch_size: {self.step_batch_size}\n"
            f"\tstep_per_device_batch_size: {self.step_per_device_batch_size}\n"
            f"\tworld_size: {world_size}\n",
        )
        assert (
            self.step_batch_size // self.step_per_device_batch_size
        ) % world_size == 0
        self.gradient_accumulation_steps = (
            self.step_batch_size // self.step_per_device_batch_size // world_size
        )

        logger.warning(
            f"Accumulation steps:\n"
            f"\trollout_accumulation_steps: {self.rollout_accumulation_steps}\n"
            f"\tgradient_accumulation_steps: {self.gradient_accumulation_steps}\n"
        )

        if self.save_steps_extra is not None:
            self.save_steps_extra_list = [
                int(string) for string in self.save_steps_extra.split("__")
            ]
        else:
            self.save_steps_extra_list = []

    def set_truncate_token_ids(self, tokenizer: transformers.PreTrainedTokenizer):
        """Convert truncation token to token ids.

        This is called in RLTrainer.
        """
        truncate_tokens = self.truncate_tokens
        if truncate_tokens is None:
            truncate_token_ids = None
        else:
            truncate_token_ids = tokenizer.convert_tokens_to_ids(truncate_tokens)
        self.truncate_token_ids = truncate_token_ids


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    training_args.data_config = data_args

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if checkpoint_dir is None and args.resume_dir is not None:
        checkpoint_dir, _ = get_last_checkpoint(args.resume_dir)
        completed_training = False

    if completed_training:
        rank0_print("Detected that training was already completed!")

    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)

    accelerator = AlpacaAccelerator(
        log_with=args.report_to,
        project_dir=args.logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        # Value model might not use all parameters (e.g., lm-head) in the forward pass.
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.ddp_find_unused_parameters,
            )
        ],
    )
    dict_args = vars(args)
    # value should be one of int, float, str, bool, or torch.Tensor
    for k in dict_args:
        if type(dict_args[k]) not in [int, float, str, bool, torch.Tensor]:
            dict_args[k] = str(dict_args[k])
    # print(dict_args)
    accelerator.init_trackers(
        "rl_trainer",
        config=dict_args,
    )
    logger.warning(
        accelerator.state,
        # main_process_only=False,
    )

    tokenizer_model_name = args.base_model_name
    TokenizerClass = AutoTokenizer

    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        with DisableLogger():
            base_model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.base_model_name,
                cache_dir=training_args.cache_dir,
            )

        vision_tower = base_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        del base_model

        if model_args.reward_base_model_name is None:
            model_args.reward_base_model_name = model_args.base_model_name
            data_args.reward_image_processor = vision_tower.image_processor
        else:
            with DisableLogger():
                reward_base_model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.reward_base_model_name,
                    cache_dir=training_args.cache_dir,
                )
            reward_vision_tower = reward_base_model.get_vision_tower()
            if not reward_vision_tower.is_loaded:
                reward_vision_tower.load_model()
            data_args.reward_image_processor = reward_vision_tower.image_processor
            del reward_base_model

        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end

    # Dataset
    data_module: dict = make_rl_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    if accelerator.is_main_process:
        training_data = data_module["train_dataset"]
        for i in range(3):
            ex_input_ids_0 = training_data[i]["queries"]
            rank0_print(ex_input_ids_0[ex_input_ids_0 != tokenizer.pad_token_id])
            ex_input_ids_0[ex_input_ids_0 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            rank0_print(tokenizer.decode(ex_input_ids_0, skip_special_tokens=False))
            rank0_print("=" * 20)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    node_id = rank // torch.cuda.device_count()

    print(f"Distributed info: rank={rank}, world_size={world_size}, node_id={node_id}")
    model_module = make_models(
        tokenizer=tokenizer,
        args=args,
        accelerator=accelerator,
        resume_from_checkpoint=(
            checkpoint_dir if training_args.resume_from_training else None
        ),
    )

    trainer = PPOTrainer(
        args=training_args,
        accelerator=accelerator,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )

    trainer.train(
        resume_training_ckpt=checkpoint_dir
        if training_args.resume_from_training
        else None
    )


if __name__ == "__main__":
    train()
