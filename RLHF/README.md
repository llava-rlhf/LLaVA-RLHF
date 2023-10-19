# (Factually Augmented) RL from Human Feedback

This RLHF codebase is mainly adapted from the [SALMON](https://github.com/Edward-Sun/SALMON) codebase, which is adapted from [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) and [QLoRA](https://github.com/artidoro/qlora).

## 0. Setup

Please refer to [`llava_setup`](../llava_setup) for instructions on how to set up the customized llava package.

Additionally, you **should** run the following command to make sure the versions of some essential packages are correct:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed==0.9.3
pip install peft==0.4.0
pip install transformers==4.31.0
pip install bitsandbytes==0.41.0
pip install datasets
```

**Note:** please install Pytorch 2.0.1 following the guidelines [here](https://pytorch.org/get-started/previous-versions/#v201). We found that the flash-attention implementation in the newest Pytorch Stable (2.1.0) could lead to buggy results. The codebase is tested with `torch==2.0.1+cu118`.

## 1. Training the Instruction-Following Reward Model

We first train an [instruction-following reward model](https://arxiv.org/abs/2310.05910) based on the [following judging creteria](prompts/reward_prompt.txt):

```text
1. Accurate: The AI should provide factual and accurate information from the image, and refrain from making statements that are not supported by the image or inconsistent with the image.
2. Helpful: The AI’s response should precisely serve the user's needs and interests, while grounding the response in the image.
3. Language Natural: The AI should employ language that flows smoothly and is free from repetitive or awkward constructs.
4. Concise: The AI should efficiently address the task or answer the question, communicating the necessary information with brevity and clarity.
```

After downloading the SFT model checkpoint from [`LLaVA-RLHF-13b-v1.5-336`](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336), the human preference data from [`LLaVA-Human-Preference-10K`](https://huggingface.co/datasets/zhiqings/LLaVA-Human-Preference-10K), and the image captions from [`LLaVA-RLHF-Data/image_to_caption.json`](https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/tree/main), you can run the training script for the reward model:

```bash
bash scripts/13b-v1.5-336/train_reward_model.sh
```

**Note**: For both 7b and 13b policy models, we use the same 13b reward model. We also provide the pretrained reward model checkpoint at [`LLaVA-RLHF-13b-v1.5-336/rm_lora_adapter_model`](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336/tree/main/rm_lora_adapter_model). To use the pretrained LoRA checkpoint, the `base_model_name_or_path` in [adapter_config.json](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336/blob/main/rm_lora_adapter_model/adapter_config.json) need to be modified to the actual path of the [SFT model](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336/tree/main/sft_model).

## 2. Initialize the RL Model

We initialize the LoRA weights of the policy model by fine-tuning the SFT model for one epoch on the combination of:

1. Our preference modeling split of the LLaVA data (10k)
2. A-OKVQA in the CoT format (5k)

We provide the processed data in [`LLaVA-RLHF-Data/llava_reward10k-aokvqa5k.json`](https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/tree/main). After downloading the data (and potentially the 7b SFT model checkpoint from [`LLaVA-RLHF-7b-v1.5-224`](https://huggingface.co/zhiqings/LLaVA-RLHF-7b-v1.5-224)), you can run the following script to initialize the policy model:

```bash
bash scripts/7b-v1.5-224/initialize_policy_model.sh
bash scripts/13b-v1.5-336/initialize_policy_model.sh
```

## 3. Training the RL Model with PPO

The PPO training of the policy model is based on the prompt combination of:

1. Our RL split of the LLaVA data (50k)
2. A-OKVQA in the CoT format (12k)
3. Yes/No Questions from VQA-v2 (10k)

We provide the processed data in [`LLaVA-RLHF-Data/llava_ppo50k-aokvqa12k-vqa10k.json`](https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/tree/main). After downloading the data, you can run the following script to train the RL model:

```bash
bash scripts/7b-v1.5-224/train_rl_model.sh
bash scripts/13b-v1.5-336/train_rl_model.sh
```
