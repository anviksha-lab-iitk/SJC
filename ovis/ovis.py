import torch 
from PIL import Image
from transformers import AutoModelForCausalLM
from pdf2image import convert_from_path
import pandas as pd 
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset_path = "../dataset.csv"
df = pd.read_csv(dataset_path)
print("Dataframe Shape: ", df.shape)

evaluation_data = dict()

def read_config_file():
    with open("ovis_config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    prompt = config['ovis2-16B']['prompt']
    temperature = config["ovis2-16B"]["temperature"]
    repetition_penalty = config["ovis2-16B"]["repetition_penalty"]
    top_p = config["ovis2-16B"]["top_p"]
    new_column = config["ovis2-16B"]["new_column"]

    return prompt, temperature, repetition_penalty, top_p, new_column

text_prompt,  _, repetition_penalty, _, new_column = read_config_file()
df[new_column] = ''

model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-16B",
                                             torch_dtype=torch.bfloat16, 
                                             multimodal_max_length=32768,
                                             trust_remote_code=True,
                                             cache_dir='/data/paditya',
                                             llm_attn_implementation="eager"
                                             ).cuda()

text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

max_partition = 9
query_prompt = f'<image>\n{text_prompt}'

for i in range(df.shape[0]):
    pdf_path = df.at[i, 'doc_filepath_linux']

    if '.pdf' in pdf_path.lower():
        try:
            pdf_images =  convert_from_path(pdf_path=pdf_path)

            if pdf_images: 
                prompt, input_ids, pixel_values = model.preprocess_inputs(query_prompt, pdf_images, max_partition=max_partition)
                attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
                input_ids = input_ids.unsqueeze(0).to(device=model.device)
                attention_mask = attention_mask.unsqueeze(0).to(device=model.device)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
                pixel_values = [pixel_values]

                with torch.inference_mode():
                    gen_kwargs = dict(
                        max_new_tokens = 1024,
                        do_sample = False,
                        top_p=None,
                        top_k=None,
                        temperature=None,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=model.generation_config.eos_token_id,
                        pad_token_id = text_tokenizer.pad_token_id,
                        use_cache=True
                    )

                    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
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

            
torch.save(evaluation_data, 'ovis2_16b_eval_data_prompt2.pt')
df.to_csv('ovis2_16b_eval_data_prompt2.csv', index=False, encoding='utf-8-sig')