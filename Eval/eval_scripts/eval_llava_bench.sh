# POPE Evaluation
export HF_HOME=/shared/sheng/huggingface
export XDG_CACHE_HOME=/shared/sheng/

export CUDA_VISIBLE_DEVICES=2

MODEL_BASE=LLaVA-RLHF-13b-v1.5-336/sft_model
MODEL_QLORA_BASE=LLaVA-RL-Fact-RLHF-13b-v1.5-336-lora-padding
MODEL_SUFFIX=$MODEL_QLORA_BASE

python model_vqa.py \
    --model-path ./checkpoints/${MODEL_BASE}/ \
    --use-qlora True --qlora-path ./checkpoints/${MODEL_QLORA_BASE} \
    --question-file \
    ./llava/qa90_questions.jsonl \
    --image-folder \
    ./eval_image/ \
    --answers-file \
    ./eval/llava/answer-file-${MODEL_SUFFIX}.jsonl --image_aspect_ratio pad --test-prompt ''

OPENAI_API_KEY="" python eval_gpt_review_visual.py \
    --question ./llava/qa90_questions.jsonl \
    --context ./table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    ./llava/qa90_gpt4_answer.jsonl \
    ./eval/llava/answer-file-${MODEL_SUFFIX}.jsonl \
    --rule ./table/rule.json \
    --output ./eval/llava/review/review-file-${MODEL_SUFFIX}.jsonl

python summarize_gpt_review.py -d ./eval/llava/review/ -f review-file-${MODEL_SUFFIX}.jsonl
