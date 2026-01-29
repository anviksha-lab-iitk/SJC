import torch
from pdf2image import convert_from_path
from PIL import Image
import yaml
import pandas as pd
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import os
os.makedirs('images', exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset_path = "../dataset.csv"
df = pd.read_csv(dataset_path)
print("Dataframe Shape: ", df.shape)

evaluation_data = dict()

ocr = PaddleOCR(use_angle_cls=True, lang='mr')

col_name = 'paddleocr-extracted-text'

def extract_text_from_images(pdf_images, i):
    extracted_text = []
    if len(pdf_images) == 1:
        input_image = np.array(pdf_images[0])
        result = ocr.ocr(input_image, cls=True)

        lines = result[0] if isinstance(result[0], list) else result
        
        boxes, txts, scores, clss = [], [], [], []
        for line in lines:
            if len(line) == 2:
                box, (text, score) = line
                boxes.append(box)
                txts.append(text)
                scores.append(score)
            elif len(line) == 3:
                box, (text, score), cls_info = line
                boxes.append(box)
                txts.append(text)
                scores.append(score)
                clss.append(cls_info)
            else:
                print(f"Unexpected OCR output format at index {i}: {line}")
        im_show = draw_ocr(np.array(input_image), boxes, txts, scores, font_path='/data/paditya/parjanya.mount/paddleocr/NotoSansDevanagari-VariableFont_wdth,wght.ttf')
        Image.fromarray(im_show).save(f'images/boxes_{i}.png')
        extracted_text.extend(txts) 
    return ' '.join(extracted_text)

def main():
    for i in range(df.shape[0]):
        pdf_path = df.at[i, 'doc_filepath_linux']

        try: 
            pdf_images = convert_from_path(pdf_path)

            if pdf_images:
                extracted_text = extract_text_from_images(pdf_images, i)
                print(f'Output (row {i}):\n{extracted_text}\n{"—"*40}')
                df.at[i, col_name] = extracted_text

                evaluation_data[pdf_path] = {
                    'pdf_path': pdf_path,
                    'extracted_text': extracted_text
                }
            else: 
                print(f"Skipping index {i}: PDF has no pages.")
        except Exception as e:
            print(f"Error processing file at index {i}: {e}")

    torch.save(evaluation_data, 'paddleocr.pt')
    df.to_csv('paddleocr.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()