"""
Note: This is not finished as it was requiring a lot of time to figure out the right metric. This code is heavily generated with the help of LLMs

"""

import json
import re
from Levenshtein import ratio


# --- 1. THE CONVERTER (Ground Truth Parsing) ---
def convert_gt(gt_data):
    """
    Parses the complex 'gt_parse' structure into a clean key-value dict.
    Reconstructs text by finding words inside label bounding boxes.
    """
    # 1. Safety check: Ensure we have the 'gt_parse' root
    parse = gt_data.get('gt_parse', gt_data)  # Handle if passed directly or wrapped

    # 2. Map all transcription words to their coordinates
    words = []
    transcriptions = parse.get('transcription', [])
    bboxes = parse.get('bbox', [])

    # Zip them together for easy lookup
    for text, box in zip(transcriptions, bboxes):
        words.append({
            'text': text,
            'x': box['x'], 'y': box['y'],
            'w': box['width'], 'h': box['height'],
            # Calculate center point for collision detection
            'cx': box['x'] + (box['width'] / 2),
            'cy': box['y'] + (box['height'] / 2)
        })

    # 3. Define the Mapping: Dataset Label -> Your Schema Key
    label_map = {
        "nam of the company": "company_name",
        "address of the company": "company_address",
        "address of the customer": "vendor_name",  # Adapting to your schema usage
        "telephone number": "phone_number",
        "date": "invoice_date",
        "sum": "total_amount",
        "IBAN": "iban",
        "invoice_id": "invoice_number",
        "invoice_no": "invoice_number"
    }

    cleaned_gt = {}

    # 4. Iterate through labeled regions to find text
    labels = parse.get('label', [])
    for lbl in labels:
        # Get the raw label name (e.g., "sum")
        raw_label = lbl['labels'][0]
        target_key = label_map.get(raw_label)

        # Only process if this is a field we care about
        if target_key:
            # Label Geometry
            lx, ly = lbl['x'], lbl['y']
            lw, lh = lbl['width'], lbl['height']

            # Find words whose CENTER point is inside the label box
            # (Adding a tiny buffer can help with edge cases)
            matched_words = []
            for w in words:
                if (lx <= w['cx'] <= lx + lw) and (ly <= w['cy'] <= ly + lh):
                    matched_words.append(w)

            # Sort words to reconstruct reading order: Top-to-Bottom, then Left-to-Right
            matched_words.sort(key=lambda k: (int(k['y']), int(k['x'])))

            # Join them
            full_text = " ".join([w['text'] for w in matched_words])

            # Store it (Overwrite if multiple boxes exist for same key, or append?)
            # Usually overwrite or append is fine. Here we overwrite.
            cleaned_gt[target_key] = full_text

    return cleaned_gt


# --- 2. THE EVALUATOR (Scoring) ---
def evaluate_invoice(prediction, ground_truth, log_func=print):
    """
    Compares prediction vs ground truth.
    Returns a dict with scores and a summary string.
    """
    report = {}

    # List of keys we expect to evaluate
    fields_to_check = [
        "company_name", "invoice_number", "invoice_date",
        "total_amount", "iban", "vendor_name", "company_address"
    ]

    log_func(f"{'FIELD':<20} | {'PREDICTION':<20} | {'GROUND TRUTH':<20} | {'SCORE'}")
    log_func("-" * 75)

    for field in fields_to_check:
        pred_val = prediction.get(field)
        gt_val = ground_truth.get(field)

        # CASE 1: Ground Truth is Missing (Data Gap)
        if gt_val is None:
            score = None
            status = "N/A (Missing GT)"

        # CASE 2: Both exist - perform comparison
        else:
            # Normalize: Lowercase, remove spaces/symbols for fairer comparison
            # (e.g. "1.00 €" vs "1.00")
            s1 = normalize_string(str(pred_val))
            s2 = normalize_string(str(gt_val))

            acc = ratio(s1, s2)
            score = round(acc, 2)
            status = str(score)

        # Print row
        p_str = (str(pred_val)[:18] + '..') if pred_val and len(str(pred_val)) > 18 else str(pred_val)
        g_str = (str(gt_val)[:18] + '..') if gt_val and len(str(gt_val)) > 18 else str(gt_val)
        log_func(f"{field:<20} | {p_str:<20} | {g_str:<20} | {status}")

        report[field] = score

    return report


def normalize_string(s):
    """Simple cleaner: lowercase and alphanumeric only"""
    if not s: return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

from datasets import load_dataset
import json
from main import compile_workflow, main
import asyncio
import io
import os
import time

DATASET_ID = "Aoschu/German_invoices_dataset_for_donut"
def clear_gt():
    output_file = "cleaned_ground_truth.json"
    all_data = []

    # Load existing data to prevent overwriting
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                all_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # Start fresh if file is corrupt or empty

    dataset = load_dataset(DATASET_ID)
    train_data = dataset['train']
    for i in range(len(train_data)):

        gt = json.loads(train_data[i]['ground_truth'])
        cleaned_data = convert_gt(gt)
        all_data.append(cleaned_data)

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)


def run_evaluation():
    pred_file = "approved_invoices_donut.json"
    gt_file = "cleaned_ground_truth.json"

    if not os.path.exists(pred_file) or not os.path.exists(gt_file):
        print(f"Files not found: {pred_file} or {gt_file}")
        return

    with open(pred_file, "r") as f:
        predictions = json.load(f)

    with open(gt_file, "r") as f:
        ground_truths = json.load(f)

    report_file = "evaluation_report.txt"
    with open(report_file, "w") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log(f"Loaded {len(predictions)} predictions and {len(ground_truths)} ground truths.")

        # Accumulators for average calculation
        field_scores = {}
        field_counts = {}

        # We assume index alignment since GT has no filename keys
        num_items = min(len(predictions), len(ground_truths))

        for i in range(num_items):
            pred_raw = predictions[i]
            gt_row = ground_truths[i]

            # Flatten the prediction dictionary: {key: {value: "val", ...}} -> {key: "val"}
            pred_flat = {}
            for key, val in pred_raw.items():
                if isinstance(val, dict) and "value" in val:
                    pred_flat[key] = val["value"]
                else:
                    pred_flat[key] = val

            log(f"\n--- Evaluation for Item {i} ---")
            if 'filename' in pred_flat:
                log(f"Filename: {pred_flat['filename']}")

            # Use the existing evaluate_invoice function
            report = evaluate_invoice(pred_flat, gt_row, log_func=log)

            # Aggregate scores
            for field, score in report.items():
                if score is not None:
                    field_scores[field] = field_scores.get(field, 0.0) + score
                    field_counts[field] = field_counts.get(field, 0) + 1

        log("\n" + "=" * 55)
        log(f"{'FIELD':<20} | {'SAMPLES':<10} | {'AVG SCORE':<10}")
        log("=" * 55)

        field_averages = []

        for field in sorted(field_scores.keys()):
            count = field_counts[field]
            avg = field_scores[field] / count
            log(f"{field:<20} | {count:<10} | {avg:.2f}")
            field_averages.append(avg)

        log("-" * 55)
        if field_averages:
            global_score = sum(field_averages) / len(field_averages)
            log(f"{'GLOBAL SCORE':<20} | {'(Macro)':<10} | {global_score:.2f}")
        log("=" * 55)


if __name__ == "__main__":
    # clear_gt()
    run_evaluation()