# (Factually Augmented) Reinforcement Learning from Human Feedback

This RLHF codebase is mainly adapted from the [SALMON](https://github.com/Edward-Sun/SALMON) codebase, which is adapted from [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) and [QLoRA](https://github.com/artidoro/qlora).

## Setup

Please refer to [`llava_setup`](../llava_setup) for instructions on how to setup the customized llava package.

Additionally, you can run the following command to make sure the versions of some essential packages are correct:

```bash
pip install "torch>=2.0.0"
pip install deepspeed==0.9.3
pip install peft==0.4.0
pip install transformers==4.33.0
pip install bitsandbytes==0.41.0
```

## Training the Instruction-Following Reward Model

We first train an [instruction-following reward model](https://arxiv.org/abs/2310.05910) based on the [following judging creteria](prompts/reward_prompt.txt):

```text
1. Accurate: The AI should provide factual and accurate information from the image, and refrain from making statements that are not supported by the image or inconsistent with the image.
2. Helpful: The AI’s response should precisely serve the user's needs and interests, while grounding the response in the image.
3. Language Natural: The AI should employ language that flows smoothly and is free from repetitive or awkward constructs.
4. Concise: The AI should efficiently address the task or answer the question, communicating the necessary information with brevity and clarity.
```

After downloading the SFT model checkpoint from [`LLaVA-RLHF-13b-v1.5-336`](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336), the human preference data from [`LLaVA-Human-Preference-10K`](https://huggingface.co/datasets/zhiqings/LLaVA-Human-Preference-10K), and the image captions from [`LLaVA-RLHF-Data/image_to_caption.json`](https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/tree/main), you can run training script for the reward model:

```bash
bash scripts/13b-v1.5-336/train_reward_model.sh
```

**Note**: For both 7b and 13b policy models, we use a 13b reward model.

## Initialize the RL Model

We initialize the LoRA weights of the policy model by fine-tuining the SFT model for one epoch on the combination of:

1. Our preference modeling split of the LLaVA data (10k)
2. A-OKVQA in the CoT fromat (5k)

We provide the processed data in [`LLaVA-RLHF-Data/llava_reward10k-aokvqa5k.json`](https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/tree/main). After downloading the data (and potentially the 7b SFT model checkpoint from [`LLaVA-RLHF-7b-v1.5-224`](https://huggingface.co/zhiqings/LLaVA-RLHF-7b-v1.5-224)), you can run the following script to initialize the policy model:

```bash
bash scripts/7b-v1.5-224/initialize_policy_model.sh
bash scripts/13b-v1.5-336/initialize_policy_model.sh
```

## Training the RL Model with PPO

The PPO training of the policy model is based on the prompt combination of:

1. Our RL split of the LLaVA data (50k)
2. A-OKVQA in the CoT format (12k)
3. Yes/No Questions from VQA-v2 (10k)

We provide the processed data in [`LLaVA-RLHF-Data/llava_ppo50k-aokvqa12k-vqa10k.json`](https://huggingface.co/datasets/zhiqings/LLaVA-RLHF-Data/tree/main). After downloading the data, you can run the following script to train the RL model:

```bash
bash scripts/7b-v1.5-224/train_rl_model.sh
bash scripts/13b-v1.5-336/train_rl_model.sh
```
