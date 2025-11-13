import os
from pathlib import Path
import json
import numpy as np
import torch
import evaluate
import argparse
import csv
import re
from transformers import Wav2Vec2ForCTC, AutoProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer

import params
from utils import load_dataset_for_ASR_without_prepare

DEVICE = torch.device("cuda")
model_id = "facebook/mms-1b-fl102"
parser = argparse.ArgumentParser(description="Evaluation with ASR.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("output_file", metavar="output-file", type=Path, help="path to output file.")
args = parser.parse_args()
output_file = Path(args.output_file)
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
model = model.to(DEVICE)
if params.lang == "en" :
    processor.tokenizer.set_target_lang("eng")
    model.load_adapter("eng")
    mapping_path = os.path.join(os.path.dirname("imports/"), "english.json")
    english_spelling_mapping = json.load(open(mapping_path))
    normalizer = EnglishTextNormalizer(english_spelling_mapping)
else :
    processor.tokenizer.set_target_lang("hun")
    model.load_adapter("hun")
    normalizer = BasicTextNormalizer()


#dataset_train = load_UASpeech_dataset(params.TRAIN_SPEAKERS, fn_kwargs)
#dataset_test = load_UASpeech_dataset(params.TEST_SPEAKERS, fn_kwargs)
dataset_testds = load_dataset_for_ASR_without_prepare(params.SZEGEDYS, params.TEST_DYSARTHRIC_SPEAKERS, args.wav_dir, True)
#test_loader = torch.utils.data.DataLoader(dataset, batch_size=params.per_device_train_batch_size)

# Group files for concatenation
file_groups = {}
ungrouped_records = []

# Process all records and separate into groups and ungrouped
for test_record in dataset_testds:
    file_path = Path(test_record['audio']['path'])
    name = file_path.stem
    # Match pattern like M001_09_1, CF014_09_01, etc. (has number at the end after underscores)
    matcher = re.match(r'^([A-Z]+\d+_\d+)_(\d+)$', name)
    if matcher:
        # Additional check: make sure it's not a complex filename with descriptive text
        prefix = matcher.group(1)  # e.g., "M001_09", "CF014_09"
        underscore_count = name.count('_')
        if prefix not in file_groups:
            file_groups[prefix] = []
        file_groups[prefix].append(test_record)
    else:
        ungrouped_records.append(test_record)

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")
all_wer_per_class = {}
all_wN_per_class = {}
all_cer_per_class = {}
all_cN_per_class = {}
average_wer_per_class = []
average_cer_per_class = []
all_transcriptions_str_per_class = {}
all_expects_str_per_class = {}
for i in range(params.label_count[params.SZEGEDYS]) :
    all_wer_per_class[i] = []
    all_wN_per_class[i] = []
    all_cer_per_class[i] = []
    all_cN_per_class[i] = []
    all_transcriptions_str_per_class[i] = ""
    all_expects_str_per_class[i] = ""
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
    # Process grouped files
    for prefix, group_records in file_groups.items():
        # Sort by sequence number
        group_records.sort(key=lambda x: int(re.search(r'_(\d+)$', Path(x['audio']['path']).stem).group(1)))
        concatenated_pred_str = ""
        lab = group_records[0]['severity']  # Assume same severity for all in group
        reference_text = group_records[0]["sentence"]
        path = Path(group_records[0]['audio']['path'])
        stem = path.stem
        base_name = re.sub(r'_\d+$', '', stem)
        modified_path = path.parent / f"{base_name}.wav"
        for record in group_records:
            audio_array = record["audio"]["array"]
            inputs = processor(audio_array, sampling_rate=params.SAMPLING_RATE, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs).logits

            ids = torch.argmax(outputs, dim=-1)[0]
            pred_str = processor.decode(ids)
            pred_str = normalizer(pred_str)
            concatenated_pred_str += " " + pred_str
        pred_str = concatenated_pred_str.strip()
        label_str = normalizer(reference_text)

        char_N = len(pred_str.replace(" ", ""))
        word_N = len(pred_str.split())

        all_transcriptions_str_per_class[lab] += " " + pred_str
        all_expects_str_per_class[lab] += " " + label_str

        wer = metric_wer.compute(predictions=[pred_str], references=[label_str])
        cer = metric_cer.compute(predictions=[pred_str], references=[label_str])

        # Store the WERs for the current batch
        all_wer_per_class[lab].extend([wer])  # Add the current batch's WERs to the list
        all_wN_per_class[lab].extend([word_N])  # Add the current batch's WERs to the list
        all_cer_per_class[lab].extend([cer])  # Add the current batch's CERs to the list
        all_cN_per_class[lab].extend([char_N])  # Add the current batch's CERs to the list
        average_wer_per_class[lab] = np.mean(all_wer_per_class[lab])
        average_cer_per_class[lab] = np.mean(all_cer_per_class[lab])

        writer.writerow([
            modified_path,
            lab,
            label_str,
            pred_str,
            wer,
            cer,
            word_N,
            char_N
        ])
    for record in ungrouped_records:
        audio_array = record["audio"]["array"]
        reference_text = record["sentence"]
        lab = record["severity"]
        inputs = processor(audio_array, sampling_rate=params.SAMPLING_RATE, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        pred_str = processor.decode(ids)
        pred_str = normalizer(pred_str)
        label_str = normalizer(reference_text)

        char_N = len(pred_str.replace(" ", ""))
        word_N = len(pred_str.split())

        all_transcriptions_str_per_class[lab] += " " + pred_str
        all_expects_str_per_class[lab] += " " + label_str

        wer = metric_wer.compute(predictions=[pred_str], references=[label_str])
        cer = metric_cer.compute(predictions=[pred_str], references=[label_str])

        # Store the WERs for the current batch
        all_wer_per_class[lab].extend([wer])  # Add the current batch's WERs to the list
        all_wN_per_class[lab].extend([word_N])  # Add the current batch's WERs to the list
        all_cer_per_class[lab].extend([cer])  # Add the current batch's CERs to the list
        all_cN_per_class[lab].extend([char_N])  # Add the current batch's CERs to the list
        average_wer_per_class[lab] = np.mean(all_wer_per_class[lab])
        average_cer_per_class[lab] = np.mean(all_cer_per_class[lab])

        writer.writerow([
            record['audio']['path'],
            lab,
            label_str,
            pred_str,
            wer,
            cer,
            word_N,
            char_N
        ])

wer_a_list = []
wer_w_list = []
cer_a_list = []
cer_w_list = []
for lab in range(params.label_count[params.SZEGEDYS]) :
    if all_transcriptions_str_per_class[lab] != "" :
        wer_a_list.append(metric_wer.compute(predictions=[all_transcriptions_str_per_class[lab]], references=[all_expects_str_per_class[lab]]))
        cer_a_list.append(metric_cer.compute(predictions=[all_transcriptions_str_per_class[lab]], references=[all_expects_str_per_class[lab]]))
        np_all_wer_per_lab = np.array(all_wer_per_class[lab])
        np_all_wN_per_lab = np.array(all_wN_per_class[lab])
        np_all_cer_per_lab = np.array(all_cer_per_class[lab])
        np_all_cN_per_lab = np.array(all_cN_per_class[lab])
        wer_w_list.append(np.average(np_all_wer_per_lab, weights=np_all_wN_per_lab))
        cer_w_list.append(np.average(np_all_cer_per_lab, weights=np_all_cN_per_lab))
    else :
        wer_a_list.append(-1.0)
        wer_w_list.append(-1.0)
        cer_a_list.append(-1.0)
        cer_w_list.append(-1.0)
print("average_wer_per_class")
print(*wer_w_list, sep='\n')
print("average_cer_per_class")
print(*cer_w_list, sep='\n')