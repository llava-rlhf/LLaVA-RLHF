### Overview

MMHal-Bench is a new evaluation benchmark specifically designed for hallucintation in Large Multimodal Models (LMM). It contains 96 challenging questions based on images from OpenImages, and their corresponding ground-truth answers and image contents.

Please check the dataset on Hugging Face (https://huggingface.co/datasets/Shengcao1006/MMHal-Bench) to download it. Here we provide the evaluation code for model generated responses.

### Usage

To evaluate your own model on MMHal-Bench, first generate model responses to the image-question pairs. You may check the template `get_response.py` about how to read and write to the response file.

After that, you may let GPT-4 rate your model's responses automatically. You will need package `openai` installed and an API key. Then, run `eval_gpt4.py`:

```
python eval_gpt4.py \
    --response [JSON file with model responses] \
    --evaluation [JSON file with GPT-4 evaluation to be saved] \
    --api-key [your OpenAI API key, starting with 'sk-'] \
    --gpt-model [GPT model to be used, or 'gpt-4-0314' by default]
```

Please note that the GPT-4 API calls are not free. Depending on your model response lengths, evaluating each question may use 1.5k-2k tokens. Also, GPT-4 responses are not deterministic, so you may get different results with the same responses.

At the end of the outputs, you can see the evaluation results like this:

```
Average score: 2.05
Hallucination rate: 0.61
Average score for each question type: 2.33,1.25,2,2.5,1.5,3.33,2.33,1.17
```