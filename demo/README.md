# Demo

This is a minimal example to launch a LLaVA-RLHF demo. In order to download the model checkpoint, please check this [Hugging Face model hub link](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336).

## Install LLaVA

To run our demo, you need to install the LLaVA package. Please follow the instructions in the [original repository](https://github.com/haotian-liu/LLaVA/tree/main#install) to install LLaVA.

## Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server *ONCE*.

### Launch a controller

```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

### Launch a gradio web server

```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

### Launch a model worker

This is the actual *worker* that performs the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```Shell
export CUDA_VISIBLE_DEVICES=0

python -m model_worker --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --load-bf16 \
    --model-name llava-rlhf-13b-v1.5-336 \
    --model-path /path/to/LLaVA-RLHF-13b-v1.5-336/sft_model \
    --lora-path /path/to/LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.

```Shell
export CUDA_VISIBLE_DEVICES=1

python -m model_worker --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port <different from 40000, say 40001> \
    --worker http://localhost:<change accordingly, i.e. 40001> \
    --load-bf16 \
    --model-name llava-rlhf-13b-v1.5-336 \
    --model-path /path/to/LLaVA-RLHF-13b-v1.5-336/sft_model \
    --lora-path /path/to/LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model
```

If you are using an Apple device with an M1 or M2 chip, you can specify the mps device by using the `--device` flag: `--device mps`.

### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs. Our latest code base will automatically try to use multiple GPUs if you have more than one GPU. You can specify which GPUs to use with `CUDA_VISIBLE_DEVICES`. Below is an example of running with the first two GPUs.

```Shell
export CUDA_VISIBLE_DEVICES=0,1

python -m model_worker --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --load-bf16 \
    --model-name llava-rlhf-13b-v1.5-336 \
    --model-path /path/to/LLaVA-RLHF-13b-v1.5-336/sft_model \
    --lora-path /path/to/LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model
```
