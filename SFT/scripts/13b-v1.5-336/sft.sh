#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
export HF_HOME=/shared/sheng/huggingface

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION=vicuna-13b-v1.5-336
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

LM_MODEL_CKPT=lmsys/vicuna-13b-v1.5
MM_CKPT=/shared/llava-$MODEL_VERSION-pretrain/mm_projector.bin
DATA_PATH=mix-llava-sft90k-vqav2_83k-okvqa_16k-flickr_23k.json


deepspeed train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LM_MODEL_CKPT \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --image_folder /shared/group/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $MM_CKPT \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-sft-llava-sft90k-vqav2_83k-okvqa_16k-pad-ep3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --image_aspect_ratio 'pad'
