from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from pathlib import Path
import argparse
import evaluate
import numpy as np
import csv

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from utils import get_LaciControl_as_list, get_LaciDys_as_list, get_Bea_as_list, get_HunDys_as_list
import params

parser = argparse.ArgumentParser(description="Evaluation with BEAST2 ASR.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("output_file", metavar="output-file", type=Path, help="path to output file.")
args = parser.parse_args()

if args.dataset == params.LACICON:
    file_paths, texts, labels = get_LaciControl_as_list(args.wav_dir)
elif args.dataset == params.LACIDYS:
    file_paths, texts, labels = get_LaciDys_as_list(args.wav_dir)
elif args.dataset == params.BEA:
    file_paths, texts, labels = get_Bea_as_list(args.wav_dir)
elif args.dataset == params.HUNDYS:
    file_paths, texts, labels = get_HunDys_as_list(args.wav_dir)

file_to_info = {str(fp): (txt, lbl) for fp, txt, lbl in zip(file_paths, texts, labels)}
normalizer = BasicTextNormalizer()

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")
# Open output CSV file
output_file = Path(args.output_file)
# ----------------------------
# STEP 1: Load existing results (if file exists)
# ----------------------------
existing_results = {}
if output_file.exists():
    with open(output_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_results[row["file_path"]] = row
    print(f"Loaded {len(existing_results)} existing results from {output_file}")
else:
    print("No existing results. Starting fresh.")
# ----------------------------
# STEP 2: Determine which files need processing
# ----------------------------
to_process = []
for i, fp in enumerate(file_paths):
    fp_str = str(fp)
    existing = existing_results.get(fp_str)
    if existing is None or existing["transcription"] == "TIMEOUT":
        to_process.append(fp)
print(f"Files to process/retry: {len(to_process)}")

# ----------------------------
# STEP 3: Process missing or failed files
# ----------------------------
if to_process:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://phon.nytud.hu/beast2")
        for audio_file_name in to_process:
            audio_file = str(audio_file_name)
            label_str_orig, lab = file_to_info[audio_file]
            label_str = normalizer(label_str_orig)

            pred_str = None
            success = False

            for attempt in range(2):  # Try up to 2 times
                try:
                    # 1. Clear if needed (optional)
                    page.click("#component-4")
                    # 2. Upload
                    page.set_input_files("#component-2 input[type='file']", audio_file)
                    # 3. Wait for waveform/duration → upload complete
                    page.wait_for_selector("#component-2 .waveform-container", timeout=600000)
                    # 4. Now click Run — safe!
                    page.click("#component-5")
                    # Wait until the textarea has non-empty value
                    page.wait_for_function("""
                        () => {
                            const textarea = document.querySelector('#component-10 textarea');
                            return textarea && textarea.value.trim() !== '' && !textarea.value.match(/^\\d+\\.\\d+s$/);
                        }
                    """, timeout=600000)
                    # Extract the actual value from the textarea
                    result = page.eval_on_selector("#component-10 textarea", "el => el.value")
                    pred_str = normalizer(result.strip())
                    success = True
                    break  # Exit retry loop on success

                except PlaywrightTimeoutError:
                    print(f"Timeout on {audio_file}, attempt {attempt + 1}/2")
                    if attempt == 0:
                        # First timeout: retry
                        continue
                    else:
                        # Second timeout: give up
                        print(f"Skipping {audio_file} after 2 timeouts.")
                        break

            # Compute metrics if successful
            if success and pred_str is not None:
                word_N = len(pred_str.split())
                char_N = len(pred_str.replace(" ", ""))
                wer = metric_wer.compute(predictions=[pred_str], references=[label_str])
                cer = metric_cer.compute(predictions=[pred_str], references=[label_str])
            else:
                pred_str = "[TIMEOUT]"
                wer = cer = -1.0
                word_N = char_N = 0

            # Update in-memory results
            existing_results[audio_file] = {
                "file_path": audio_file,
                "class_label": str(lab),
                "expected_text": label_str,
                "transcription": pred_str,
                "wer": str(wer),
                "cer": str(cer),
                "word_count": str(word_N),
                "char_count": str(char_N)
            }
# ----------------------------
# STEP 4: OVERWRITE the original file with FULL updated results
# ----------------------------
all_wer_per_class = {}
all_wN_per_class = {}
all_cer_per_class = {}
all_cN_per_class = {}
average_wer_per_class = []
average_cer_per_class = []
for i in range(params.label_count[params.dataset]):
    all_wer_per_class[i] = []
    all_wN_per_class[i] = []
    all_cer_per_class[i] = []
    all_cN_per_class[i] = []
    average_wer_per_class.append(-1)
    average_cer_per_class.append(-1)
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "file_path",
        "class_label",
        "expected_text",
        "transcription",
        "wer",
        "cer",
        "word_count",
        "char_count"
    ])
    # Write in original file order
    for fp in file_paths:
        fp_str = str(fp)
        row = existing_results.get(fp_str)
        if row is None:
            # Fallback (shouldn't happen if logic is correct)
            writer.writerow([fp_str, "", "", "[MISSING]", -1, -1, 0, 0, "missing"])
        else:
            writer.writerow([
                row["file_path"],
                row["class_label"],
                row["expected_text"],
                row["transcription"],
                row["wer"],
                row["cer"],
                row["word_count"],
                row["char_count"]
            ])
            all_wer_per_class[row["class_label"]].extend([row["wer"]])  # Add the current batch's WERs to the list
            all_wN_per_class[row["class_label"]].extend([row["word_count"]])  # Add the current batch's WERs to the list
            all_cer_per_class[row["class_label"]].extend([row["cer"]])  # Add the current batch's CERs to the list
            all_cN_per_class[row["class_label"]].extend([row["char_count"]])  # Add the current batch's CERs to the list
            average_wer_per_class[row["class_label"]] = np.mean(all_wer_per_class[row["class_label"]])
            average_cer_per_class[row["class_label"]] = np.mean(all_cer_per_class[row["class_label"]])

wer_w_list = []
cer_w_list = []
for lab in range(params.label_count[params.dataset]) :
    if len(all_wN_per_class[lab])>0 :
        np_all_wer_per_lab = np.array(all_wer_per_class[lab])
        np_all_wN_per_lab = np.array(all_wN_per_class[lab])
        np_all_cer_per_lab = np.array(all_cer_per_class[lab])
        np_all_cN_per_lab = np.array(all_cN_per_class[lab])
        wer_w_list.append(np.average(np_all_wer_per_lab, weights=np_all_wN_per_lab))
        cer_w_list.append(np.average(np_all_cer_per_lab, weights=np_all_cN_per_lab))
    else :
        wer_w_list.append(-1.0)
        cer_w_list.append(-1.0)
print(wer_w_list, cer_w_list)
print("final average_wer_per_class")
print(*average_wer_per_class, sep='\n')
print("final average_cer_per_class")
print(*average_cer_per_class, sep='\n')
print("average_wer_per_class")
print(*wer_w_list, sep='\n')
print("average_cer_per_class")
print(*cer_w_list, sep='\n')