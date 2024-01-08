import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
import json
from PIL import Image
from tqdm import tqdm

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import IdeficsForVisionText2Text, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

if __name__ == "__main__":
    
    checkpoint = "HuggingFaceM4/idefics-80b-instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, 
                                                      quantization_config=quantization_config, 
                                                      device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    nejms = sorted(glob.glob('../../Datasets/NEJM/NEJM/*/*.png'))
    answers = []  
    for n in tqdm(range(2, len(nejms))):
        image_paths = [nejms[0], nejms[1], nejms[n]]
        text_paths = [os.path.join(os.path.dirname(path), 'content.txt') for path in image_paths]
        images = [Image.open(path) for path in image_paths]
        texts = [open(path).read() for path in text_paths]
        
        lst = texts[0].split('\n')[1:8]
        lst.append('Assistant: ' + texts[0].split('\n')[8][8:] + '.\n')
        example = '\n'.join(lst)

        lst = texts[1].split('\n')[1:8]
        lst.append('Assistant: ' + texts[1].split('\n')[8][8:] + '.\n')
        example_ = '\n'.join(lst)

        lst = texts[2].split('\n')[1:8]
        lst.append('Assistant:')
        question = '\n'.join(lst)
        texts_ = [example, example_, question]

        prompts = [
            [
                "User:",
                images[0],
                texts_[0],
                "User:",
                images[1],
                texts_[1],
                "User:",
                images[2],
                texts_[2],
            ]
        ]

        inputs = processor(prompts[0], return_tensors="pt").to(device)

        decode_config = {
                "temperature": 0.0,
                "do_sample": False,
                "max_length": 2048,
            }
        
        answer = []
        generated_ids = model.generate(**inputs, **decode_config)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        answer.append(generated_text[0].split('\n')[23].split('User')[0][11:].strip('.'))

        result = {
            "id": os.path.dirname(image_paths[-1]).split('/')[-1],
            "question": question,
            "model answer": answer,
        }
        answers.append(result)

        #### save the results ####
        with open('nejm_80B.json', 'w') as f:
            json.dump(answers, f, indent=4)
    