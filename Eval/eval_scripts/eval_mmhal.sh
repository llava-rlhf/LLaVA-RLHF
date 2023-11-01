# MMHal-Bench Evaluation
export HF_HOME=/shared/sheng/huggingface
export XDG_CACHE_HOME=/shared/sheng/

export CUDA_VISIBLE_DEVICES=2

MODEL_BASE=LLaVA-RLHF-13b-v1.5-336/sft_model
MODEL_QLORA_BASE=LLaVA-RL-Fact-RLHF-13b-v1.5-336-lora-padding
MODEL_SUFFIX=$MODEL_QLORA_BASE

python model_vqa_mmhal.py \
    --model-path ./checkpoints/${MODEL_BASE}/ \
    --use-qlora True --qlora-path ./checkpoints/${MODEL_QLORA_BASE} \
    --temperature 0.0 \
    --answers-file \
    ./eval/mmhal/answer-file-${MODEL_SUFFIX}.json --image_aspect_ratio pad --test-prompt ''

python eval_gpt_mmhal.py \
    --response ./eval/mmhal/answer-file-${MODEL_SUFFIX}.json \
    --evaluation ./eval/mmhal/review-file-${MODEL_SUFFIX}.json \
    --api-key "" \
    --gpt-model gpt-4-0314

python summarize_gpt_mmhal.py \
    --evaluation ./eval/mmhal/review-file-${MODEL_SUFFIX}.json
