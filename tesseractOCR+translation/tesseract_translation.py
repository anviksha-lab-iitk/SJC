import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import yaml
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

src_lang, tgt_lang = "mar_Deva", "eng_Latn"

model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager"
).to(DEVICE)

ip = IndicProcessor(inference=True)

dataset_path = "../dataset.csv"
df = pd.read_csv(dataset_path)
print("Dataframe Shape: ", df.shape)

evaluation_data = dict()

def read_config_file():
    with open("tesseract_translation_config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    tesseract_col_name = config['tesseract']['tesseract_col_name']
    translation_col_name = config['tesseract']['translation_col_name']
    return tesseract_col_name, translation_col_name

tesseract_col_name, translation_col_name = read_config_file()

df[tesseract_col_name] = ''
df[translation_col_name] = ''


def extract_text_from_images(pdf_images):
    extracted_text = ''
    if len(pdf_images) == 1:
        extracted_text = pytesseract.image_to_string(pdf_images[0], lang='mar')

    print("Extracted text :: ", extracted_text)
    return extracted_text

def translate(extracted_text):

    input_sentences = [extracted_text]
    
    batch = ip.preprocess_batch(
    input_sentences, 
    src_lang=src_lang,
    tgt_lang=tgt_lang,
    )

    inputs = tokenizer(
        batch, 
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, 
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5
        )

    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    print("Translated :: ")

    return translations

for i in range(df.shape[0]):
    pdf_path = df.at[i, 'doc_filepath_linux']

    try:
        pdf_images = convert_from_path(pdf_path=pdf_path)
        
        print('==================================================')
        print(pdf_path)
        if pdf_images:
            extracted_text = extract_text_from_images(pdf_images)
            translated_text = translate(extracted_text)
            print(f'Output (row {i}):\n{translated_text}\n{"—"*40}')
            print('==================================================')
            df.at[i, tesseract_col_name] = extracted_text
            df.at[i, translation_col_name] = translated_text

            evaluation_data[pdf_path] = {
                'pdf_path': pdf_path,
                'translation': df.at[i, 'translation'],
                'extracted_text': extracted_text,
                'translated_text': translated_text
            }

        else: 
            print(f"Skipping index {i}: PDF has no pages.")
    except Exception as e:
        print(f"Error processing file at index {i}: {e}")

torch.save(evaluation_data, 'tesseract_translation_prompt1.pt')
df.to_csv('tesseract_translation_prompt1.csv', index=False, encoding='utf-8-sig')
