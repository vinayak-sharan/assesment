"""
Some utility tools to run the hf dataset

"""

from datasets import load_dataset
from assesment.main import compile_workflow, main
import asyncio
import io
import os
import time

DATASET_ID = "Aoschu/German_invoices_dataset_for_donut"


def image_to_bytes(image_obj):
    img_byte_arr = io.BytesIO()
    image_obj.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


import json


def extract_text_from_gt(gt_data):
    """
    Converts the complex layout-based GT into a simple key-value dictionary.
    """
    parse = gt_data.get('gt_parse', {})

    # 1. Map all words to their coordinates
    # We create a list of dicts: [{'text': 'Invoice', 'x': 10, 'y': 20...}, ...]
    words = []
    transcriptions = parse.get('transcription', [])
    bboxes = parse.get('bbox', [])

    # Safety check: lengths must match
    for text, box in zip(transcriptions, bboxes):
        words.append({
            'text': text,
            'x': box['x'],
            'y': box['y'],
            'w': box['width'],
            'h': box['height']
        })

    # 2. Extract specific fields based on 'label' regions
    extracted_gt = {}
    labels = parse.get('label', [])

    for lbl in labels:
        field_name = lbl['labels'][0]  # e.g., "nam of the company"

        # Define the region of interest from the label
        lx, ly, lw, lh = lbl['x'], lbl['y'], lbl['width'], lbl['height']

        # Find all words that are predominantly INSIDE this label box
        matched_words = []
        for w in words:
            # Simple center-point check: is the center of the word inside the label box?
            w_center_x = w['x'] + (w['w'] / 2)
            w_center_y = w['y'] + (w['h'] / 2)

            if (lx <= w_center_x <= lx + lw) and (ly <= w_center_y <= ly + lh):
                matched_words.append(w)

        # Sort words by Y (lines) then X (left-to-right) to reconstruct sentences
        # (Simple sorting, might need tweaking for complex multi-column layouts)
        matched_words.sort(key=lambda k: (int(k['y']), k['x']))

        full_text = " ".join([w['text'] for w in matched_words])

        # Normalize keys to match your Pydantic model
        std_key = normalize_key(field_name)
        extracted_gt[std_key] = full_text

    return extracted_gt


def normalize_key(raw_label):
    """Maps dataset labels to your Agent's schema keys"""
    mapping = {
        "nam of the company": "company_name",
        "address of the company": "company_address",
        "address of the customer": "vendor_name",  # Verify this mapping based on context
        "telephone number": "phone_number",
        "date": "invoice_date",
        "sum": "total_amount",
        "IBAN": "iban",
        # Add others as found in the dataset
    }
    return mapping.get(raw_label, raw_label)

def run_donut_dataset():
    dataset = load_dataset(DATASET_ID)
    train_data = dataset['train']
    for i in range(len(train_data)):
        image = train_data[i]['image']
        image_bytes = image_to_bytes(image)
        # clean_json = extract_text_from_gt(json.loads(train_data[i]['ground_truth']))
        file_name = os.path.basename(json.loads(train_data[i]['ground_truth'])['gt_parse']["ocr"])
        app = compile_workflow()
        asyncio.run(main(app, None, num_agents=1, test_mode=[file_name, image_bytes]))
        time.sleep(1)

if __name__ == "__main__":
    run_donut_dataset()



