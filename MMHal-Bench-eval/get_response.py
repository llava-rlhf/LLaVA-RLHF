import argparse
import json

import requests
from PIL import Image
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='response_template.json', help='template file containing images and questions')
    parser.add_argument('--output', type=str, default='response_mymodel.json', help='output file containing model responses')
    parser.add_argument('--mymodel', type=str)
    args = parser.parse_args()

    # build the model using your own code
    mymodel = build_mymodel(args)

    json_data = json.load(open(args.input_json, 'r'))

    for idx, line in enumerate(json_data):
        image_src = line['image_src']
        image = load_image(image_src)
        question = line['question']
        response = mymodel(image, question)
        # print(idx, response)
        line['model_answer'] = response

    with open(args.output, 'w') as f:
        json.dump(json_data, f, indent=2)
