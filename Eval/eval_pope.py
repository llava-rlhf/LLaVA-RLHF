import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
import json
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init
from PIL import Image

import os
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from glob import glob


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
# image_dir = "/mnt/bn/data-tns-algo-masp/data/coco/val2017"

def divide_chunks(l, n=2):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

    return 

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images(image_folder):
    image_files = glob(os.path.join(image_folder, '*'))
    images = []
    for image_file in image_files:
        images.append(load_image(image_file))
    return images

def read_sources(source_file):
    # task_txt = "/mnt/bd/bohanzhaiv1/LLM/bohan/Awesome-Multimodal-Large-Language-Models/tools/eval_tool/LaVIN/existence.txt"
    # lines = open(task_txt, 'r').readlines()
    lines = json.load(open(source_file, 'r'))
    chunk_lines = list(lines) # one image corresponds to two questions
    return chunk_lines

def model_inference(model, tokenizer, question, image_path, image_processor):
    conv = conv_templates["multimodal"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    image = load_image(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs



def process_line(line):
    question = line['text']
    ans = line['label']
    image_name = line['image']
    return image_name, question, ans

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def main():
    disable_torch_init()
    image_dir = '/mnt/bn/algo-masp-nas-2/masp_data/coco_2014/val2014'

    model_name = "/mnt/bn/algo-masp-nas-2/weights/llava/LLaVA-13b-v1-1"
    model_type = 'llava'
    
    model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    # vision_tower.to(device='cuda', dtype=torch.float16)\

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2


    ck_lines = read_sources("/mnt/bd/bohanzhaiv1/LLM/bohan/POPE/output/coco/coco_pope_adversarial.json")
    results = []
    for i, ck_line in tqdm(enumerate(ck_lines), total=len(ck_lines)):
        image_name, question, ans = process_line(ck_line)
        rt = {'question_id':ck_line['question_id'], 'image':image_name, 'text':question}
        image_path = os.path.join(image_dir, image_name)
        qs = question
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + '\n' + qs

        if model_type == 'mpt':
            conv_mode = "mpt_multimodal"
        else:
            conv_mode = "multimodal"
            conv_mode = "vicuna_v1_1"
            # conv_mode = "v1"
            # conv_mode = "vicuna_v1"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                num_beams=1,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip().replace('\n', '')
        rt['answer'] = outputs
        results.append(rt)

    with open('/mnt/bd/bohanzhaiv1/LLM/bohan/POPE/answer/coco_pope_adversarial.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()