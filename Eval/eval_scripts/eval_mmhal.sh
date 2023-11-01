# POPE Evaluation
export HF_HOME=/shared/sheng/huggingface
export XDG_CACHE_HOME=/shared/sheng/

export CUDA_VISIBLE_DEVICES=2

MODEL_BASE=LLaVA-RLHF-13b-v1.5-336/sft_model
MODEL_QLORA_BASE=LLaVA-RL-Fact-RLHF-13b-v1.5-336-lora-padding
MODEL_SUFFIX=$MODEL_QLORA_BASE

# git lfs install
# git clone https://huggingface.co/Shengcao1006/MMHal-Bench

python model_vqa.py \
    --model-path ./checkpoints/${MODEL_BASE}/ \
    --use-qlora True --qlora-path ./checkpoints/${MODEL_QLORA_BASE} \
    --temperature 0.0 \
    --question-file \
    ./MMHal-Bench/mmhal96_questions.jsonl \
    --image-folder \
    ./MMHal-Bench/images/ \
    --answers-file \
    ./eval/mmhal/answer-file-${MODEL_SUFFIX}.jsonl --image_aspect_ratio pad --test-prompt ''

cd ../MMHal-Bench-eval

python eval_gpt4.py \
    --response [JSON file with model responses] \
    --evaluation [JSON file with GPT-4 evaluation to be saved] \
    --api-key [your OpenAI API key, starting with 'sk-'] \
    --gpt-model [GPT model to be used, or 'gpt-4-0314' by default]
