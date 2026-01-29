import os
from pathlib import Path
import json
import numpy as np
import torch
import evaluate
import argparse
import csv
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from nemo.collections.speechlm2.models import SALM
from accelerate import Accelerator
import soundfile as sf
import params
from utils import load_dataset_for_ASR_without_prepare

accelerator = Accelerator()
DEVICE = accelerator.device
CPU = torch.device('cpu')

parser = argparse.ArgumentParser(description="Evaluation using IBM's Granite Speech ASR.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("output_file", metavar="output-file", type=Path, help="path to output file.")
args = parser.parse_args()
normalizer = BasicTextNormalizer()
model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
model = model.to(DEVICE)
user_prompt = f"Transcribe the following: {model.audio_locator_tag}"
mapping_path = os.path.join(os.path.dirname("imports/"), "english.json")
english_spelling_mapping = json.load(open(mapping_path))

dataset_testds = load_dataset_for_ASR_without_prepare(args.dataset, params.TEST_DYSARTHRIC_SPEAKERS, args.wav_dir, True)

output_file = Path(args.output_file)
# ----------------------------
# STEP 1: Load existing results (if file exists)
# ----------------------------
existing_results = {}
file_is_new = not output_file.exists()
if not file_is_new:
    with open(output_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_results[Path(row["file_path"]).name] = row
    print(f"Loaded {len(existing_results)} existing results from {output_file}")
else:
    print("No existing results. Starting fresh.")

to_process = []
results = []
for j, test_record in enumerate(dataset_testds):
    fp_str = str(test_record['audio']['path'])
    existing = existing_results.get(Path(fp_str).name)
    if existing is None :
        to_process.append(test_record)
    else :
        results.append(existing)
print(f"Files to process/retry: {len(to_process)}")
print(f"Files that have already proceed: {len(results)}")
metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")
if to_process:
    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if file_is_new:
            writer.writerow(["file_path", "class_label", "expected_text", "transcription", "wer", "cer", "word_count", "char_count"])

        for idx, test_record in enumerate(to_process):
            audio_array = test_record["audio"]["array"]
            orig_sr = test_record["audio"]["sampling_rate"]
            reference_text = test_record["sentence"]
            severity = test_record["severity"]

            i = 0
            total_samples = len(audio_array)
            chunk_predictions = []
            while i < total_samples:
                start = i
                end = i + params.CHUNK_SAMPLES
                if (total_samples - end) < params.CHUNK_SAMPLES:
                    end = total_samples

                if i == 0 and end == total_samples:
                    audio_to_process = [test_record['audio']['path']]
                else:
                    chunk = audio_array[start:end]
                    temp_chunk_path = Path(f"temp_chunk_{i}.wav")
                    sf.write(temp_chunk_path, chunk, params.SAMPLE_RATE)
                    audio_to_process = [str(temp_chunk_path)]

                prompt = [{"role": "user", "content": user_prompt, "audio": audio_to_process}, ]
                answer_ids = model.generate(prompts=[prompt], max_new_tokens=128, )
                text = model.tokenizer.ids_to_text(answer_ids[0].cpu())

                text_norm_strip = normalizer(text.strip())
                chunk_predictions.append(text_norm_strip)

                if temp_chunk_path and temp_chunk_path.exists():
                    temp_chunk_path.unlink()  # This is the pathlib way to delete files
                i = end
            # Concatenate all predictions
            pred_str = " ".join(chunk_predictions).strip()
            label_str = normalizer(reference_text)

            lab = test_record['severity']

            char_N = len(label_str.replace(" ", ""))
            word_N = len(label_str.split())

            wer = metric_wer.compute(predictions=[pred_str], references=[label_str])
            cer = metric_cer.compute(predictions=[pred_str], references=[label_str])

            results.append( {
                "file_path": str(test_record['audio']['path']),
                "class_label": str(lab),
                "expected_text": label_str,
                "transcription": pred_str,
                "wer": str(wer),
                "cer": str(cer),
                "word_count": str(word_N),
                "char_count": str(char_N)
            })
            writer.writerow([
                test_record['audio']['path'],
                lab,
                label_str,
                pred_str,
                wer,
                cer,
                word_N,
                char_N
            ])
all_wer_per_class = {}
all_wN_per_class = {}
all_cer_per_class = {}
all_cN_per_class = {}
for i in range(params.label_count[args.dataset]) :
    all_wer_per_class[i] = []
    all_wN_per_class[i] = []
    all_cer_per_class[i] = []
    all_cN_per_class[i] = []
for idx, row in enumerate(results):
    lab = int(row["class_label"])
    wer = float(row["wer"])
    cer = float(row["cer"])
    word_N = len(row["expected_text"].split())
    char_N = len(row["expected_text"].replace(" ", ""))
    all_wer_per_class[lab].extend([wer])  # Add the current batch's WERs to the list
    all_wN_per_class[lab].extend([word_N])  # Add the current batch's WERs to the list
    all_cer_per_class[lab].extend([cer])  # Add the current batch's CERs to the list
    all_cN_per_class[lab].extend([char_N])  # Add the current batch's CERs to the list

wer_w_list = []
cer_w_list = []
for lab in range(params.label_count[args.dataset]) :
    if all_wer_per_class[lab]:
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
print("average_wer_per_class")
print(*wer_w_list, sep='\n')
print("average_cer_per_class")
print(*cer_w_list, sep='\n')