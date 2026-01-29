import torch
from pdf2image import convert_from_path
from PIL import Image,  ImageFont, ImageDraw
import yaml
import pandas as pd
import easyocr
import numpy as np
import cv2



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset_path = "../dataset.csv"
df = pd.read_csv(dataset_path)
print("Dataframe Shape: ", df.shape)

evaluation_data = dict()

reader = easyocr.Reader(['mr'])

col_name = 'easyocr-extracted-text'

def draw_text_with_pil(image_bgr, text, position):
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font_path = "/data/paditya/parjanya.mount/easyocr/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
    font = ImageFont.truetype(font_path, 20)

    draw.text(position, text, font=font, fill=(0, 0, 255))  # blue color

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def extract_text_from_images(pdf_images, i):
    extracted_text = ''
    if len(pdf_images) == 1:
        input_image = np.array(pdf_images[0])
        
        image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        results = reader.readtext(image_bgr)
        for result in results:
            bbox, text, confidence = result
            extracted_text += text
            pts = np.array(bbox).astype(int)
            cv2.polylines(image_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            image_bgr = draw_text_with_pil(image_bgr, text, tuple(pts[0]))
            cv2.imwrite(f'images/boxes_{i}.png', image_bgr)
    return extracted_text

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

    torch.save(evaluation_data, 'easyocr.pt')
    df.to_csv('easyocr.csv', index=False, encoding='utf-8-sig')

    

if __name__ == "__main__":
    main()