#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v0_plain
########### DO NOT CHANGE ###########

DATA_PATH=/data/blip_laion_cc_sbu_558k.json
LM_MODEL_CKPT="lmsys/vicuna-13b-v1.5"
MODEL_VERSION=vicuna-13b-v1.5-336

deepspeed train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LM_MODEL_CKPT \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder /shared/group/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /model/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --image_folder
