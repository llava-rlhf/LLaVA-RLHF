# POPE Evaluation
export HF_HOME=/shared/sheng/huggingface
export XDG_CACHE_HOME=/shared/sheng/

MMBENCH_CAT='dev'

export CUDA_VISIBLE_DEVICES=2 

MODEL_BASE=LLaVA-RLHF-13b-v1.5-336/sft_model
MODEL_QLORA_BASE=LLaVA-RL-Fact-RLHF-13b-v1.5-336-lora-padding
MODEL_SUFFIX=$MODEL_QLORA_BASE

python model_mmbench.py \
    --short_eval True \
    --model-path ./checkpoints/${MODEL_BASE}/ \
    --use-qlora True --qlora-path ./checkpoints/${MODEL_QLORA_BASE} \
    --question-file \
    ./mmbench/mmbench_${MMBENCH_CAT}_20230712.tsv \
    --image-folder \
    ./eval_image/ \
    --answers-file \
    ./eval/mmbench/answer-file-${MODEL_SUFFIX}_${MMBENCH_CAT}$.xlsx --image_aspect_ratio square --test-prompt '\nAnswer the question using a single word or phrase.'

# submit the answer file to https://opencompass.org.cn/mmbench-submission

