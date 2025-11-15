import os
from pathlib import Path
import json
import numpy as np
import torch
import evaluate
import argparse
import csv
from nemo.collections.asr.models import ASRModel
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer

import params
from utils import get_LaciControl_as_list, get_LaciDys_as_list, get_TrogoGenerated_as_list, get_HunDys_as_list, get_UASpeech_as_list, get_SzegedDys_as_list

DEVICE = torch.device("cuda")
parser = argparse.ArgumentParser(description="Evaluation with ASR.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("output_file", metavar="output-file", type=Path, help="path to output file.")
args = parser.parse_args()
output_file = Path(args.output_file)
model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
model = model.to(DEVICE)
if args.dataset in params.hungarian_datasets :
    lang = "hu"
else :
    lang = "en"
if lang== "en" :
    mapping_path = os.path.join(os.path.dirname("imports/"), "english.json")
    english_spelling_mapping = json.load(open(mapping_path))
    normalizer = EnglishTextNormalizer(english_spelling_mapping)
else :
    normalizer = BasicTextNormalizer()
dataset_dir=args.wav_dir
file_paths = []
texts = []
labels = []
if args.dataset is params.TORGO_GENERATED:
    file_paths, texts, labels = get_TrogoGenerated_as_list(params.torgo_generated_dir)
elif args.dataset is params.LACICON:
    dataset_dir_final = params.laci_control_dir if dataset_dir == Path() else dataset_dir
    file_paths, texts, labels = get_LaciControl_as_list(dataset_dir_final)
elif args.dataset is params.LACIDYS:
    dataset_dir_final = params.laci_dys_dir if dataset_dir == Path() else dataset_dir
    file_paths, texts, labels = get_LaciDys_as_list(dataset_dir_final)
elif args.dataset is params.SZEGEDYS:
    file_paths, texts, labels = get_SzegedDys_as_list(dataset_dir)
elif args.dataset is params.HUNDYS:
    file_paths, texts, labels = get_HunDys_as_list(dataset_dir)

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
for i in range(params.label_count[args.dataset]) :
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
    for idx, file_path in enumerate(file_paths):
        # Concatenate all predictions
        pred_str = normalizer(model.transcribe([file_path], source_lang='hu', target_lang='hu')[0].text)
        label_str = normalizer(texts[idx])
        lab = labels[idx]

        char_N = len(label_str.replace(" ", ""))
        word_N = len(label_str.split())

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
            file_path,
            str(lab),
            label_str,
            pred_str,
            str(wer),
            str(cer),
            str(word_N),
            str(char_N)
        ])


wer_a_list = []
wer_w_list = []
cer_a_list = []
cer_w_list = []
for lab in range(params.label_count[args.dataset]) :
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
print(wer_a_list, cer_a_list)
print(wer_w_list, cer_w_list)
print("final average_wer_per_class")
print(*average_wer_per_class, sep='\n')
print("final average_cer_per_class")
print(*average_cer_per_class, sep='\n')
print("global average_wer_per_class")
print(*wer_a_list, sep='\n')
print("global average_cer_per_class")
print(*cer_a_list, sep='\n')
print("average_wer_per_class")
print(*wer_w_list, sep='\n')
print("average_cer_per_class")
print(*cer_w_list, sep='\n')