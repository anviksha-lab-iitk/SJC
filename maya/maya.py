import torch
import os
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from llava.model.language_model.llava_cohere import LlavaCohereForCausalLM, LlavaCohereConfig
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates, SeparatorStyle
import pandas as pd
from pdf2image import convert_from_path
import yaml
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

dataset_path = '../dataset.csv'
df = pd.read_csv(dataset_path)
print("Dataframe Shape: ", df.shape)
device_map = 'cuda'
torch_dtype = torch.float16

evaluation_data = dict()

def read_config_file():
    with open("maya_config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    prompt = config['maya-8B']['prompt']
    temperature = config['maya-8B']['temperature']
    repetition_penalty = config['maya-8B']['repetition_penalty']
    top_p = config['maya-8B']['top_p']
    new_column = config['maya-8B']['new_column']

    return prompt, temperature, repetition_penalty, top_p, new_column

text_prompt, temperature, repetition_penalty, _, new_column = read_config_file()

df[new_column] = ''

config = LlavaCohereConfig.from_pretrained("maya-multimodal/maya")
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B", use_fast=True)

model = LlavaCohereForCausalLM.from_pretrained(
    "maya-multimodal/maya",
    config=config,
    device_map=device_map,
    torch_dtype=torch_dtype
)

tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
if getattr(config, "mm_use_im_start_end", False):
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map=device_map)
vision_tower.to(device=device_map, dtype=torch_dtype)
image_processor = vision_tower.image_processor

def predict(images, question:str, temperature=0.0, max_new_tokens=100):
    image_tensor = process_images([images], image_processor, config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch_dtype) for img in image_tensor]
    else: 
        image_tensor = image_tensor.to(model.device, dtype=torch_dtype)
    
    conv = conv_templates["aya"].copy()
    inp = question
    if config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt" 
    ).unsqueeze(0).to(model.device)

    outputs_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[images.size],
        do_sample=temperature > 0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    return tokenizer.decode(outputs_ids[0], skip_special_tokens=True).strip()

def main():
    for i in range(df.shape[0]):
        pdf_path = df.at[i, 'doc_filepath_linux']

        if '.pdf' in pdf_path.lower():
            try: 
                pdf_images = convert_from_path(pdf_path=pdf_path)
                
                if pdf_images:
                    output = predict(images=pdf_images[0], question=text_prompt, temperature=temperature)
                    print(f'Output (row {i}):\n{output}\n{"—"*40}')


                    evaluation_data[pdf_path] = {
                        'pdf_path': pdf_path,
                        'translation': df.at[i, 'translation'],
                        'ovis2_16B': output
                    }

                    df.at[i, new_column] = output
                else:
                    print(f"Skipping index {i}: PDF has no pages.")
            except Exception as e:
                print(f"Error processing file at index {i}: {e}")
    torch.save(evaluation_data, 'maya_8b_eval_data_prompt2.pt')
df.to_csv('maya_8b_eval_data_prompt2.csv', index=False, encoding='utf-8-sig')            

if __name__ == "__main__":
    main()

