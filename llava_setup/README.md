# Install LLaVA

We use LLaVA version `6cea223` for training the SFT and RLHF models.

## Apply the custom patch

```bash
git clone https://github.com/haotian-liu/LLaVA.git

cd LLaVA

git reset --hard 6cea223

git apply < ../fix_llava_padding.patch
```

## Install LLaVA

Next, please follow the instructions in the [original repository](https://github.com/haotian-liu/LLaVA/tree/6cea223532a7ab7bda8116336c59772faccdcbca#install) to install LLaVA.

## Update Packages

Finally, please update the following packages:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed==0.9.3
pip install peft==0.4.0
pip install transformers==4.31.0
pip install bitsandbytes==0.41.0
pip install datasets
```

**Note:** please install Pytorch 2.0.1 following the guidelines [here](https://pytorch.org/get-started/previous-versions/#v201). We found that the flash-attention implementation in the newest Pytorch Stable (2.1.0) could lead to buggy results. The codebase is tested with `torch==2.0.1+cu118`.
