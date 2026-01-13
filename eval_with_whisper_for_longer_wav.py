import os
from pathlib import Path
import json
import numpy as np
import torch
import evaluate
import argparse
import csv
from transformers import (WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer

from accelerate import Accelerator

import params
from utils import load_dataset_for_ASR_without_prepare

accelerator = Accelerator()
DEVICE = accelerator.device
CPU = torch.device('cpu')

parser = argparse.ArgumentParser(description="Evaluation with ASR.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("output_file", metavar="output-file", type=Path, help="path to output file.")
args = parser.parse_args()
output_file = Path(args.output_file)
model = WhisperForConditionalGeneration.from_pretrained(params.whisper_arch, torch_dtype=params.torch_dtype)
model = model.to(DEVICE)
processor = WhisperProcessor.from_pretrained(params.whisper_arch, torch_dtype=params.torch_dtype)
tokenizer = processor.tokenizer
task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
if args.dataset in params.hungarian_datasets :
    lang = "hu"
else :
    lang = "en"
if lang== "en" :
    lang_token_id = tokenizer.convert_tokens_to_ids("<|en|>")
    model.generation_config.language = "english"
    mapping_path = os.path.join(os.path.dirname("imports/"), "english.json")
    english_spelling_mapping = json.load(open(mapping_path))
    normalizer = EnglishTextNormalizer(english_spelling_mapping)
else :
    lang_token_id = tokenizer.convert_tokens_to_ids("<|hu|>")
    model.generation_config.language = "hungarian"
    normalizer = BasicTextNormalizer()

model.config.forced_decoder_ids = [ (1, lang_token_id), (2, task_token_id) ]
model.generation_config.task = "transcribe"
#augmentor = SpeedAugmentation(params.SAMPLING_RATE, params.speed_factor)

fn_kwargs = {"feature_extractor":  processor.feature_extractor,
             "tokenizer": processor.tokenizer,
             "augmentor": None}

#dataset_train = load_UASpeech_dataset(params.TRAIN_SPEAKERS, fn_kwargs)
#dataset_test = load_UASpeech_dataset(params.TEST_SPEAKERS, fn_kwargs)
dataset_testds = load_dataset_for_ASR_without_prepare(args.dataset, params.TEST_DYSARTHRIC_SPEAKERS, args.wav_dir, True)
#test_loader = torch.utils.data.DataLoader(dataset, batch_size=params.per_device_train_batch_size)

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
            existing_results[row["file_path"]] = row
    print(f"Loaded {len(existing_results)} existing results from {output_file}")
else:
    print("No existing results. Starting fresh.")

to_process = []
results = []
for i, test_record in enumerate(dataset_testds):
    fp_str = str(test_record['audio']['path'])
    existing = existing_results.get(fp_str)
    if existing is None :
        to_process.append(test_record)
    else :
        results.append(existing)
print(f"Files to process/retry: {len(to_process)}")
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

            total_samples = len(audio_array)
            num_chunks = int(np.ceil(total_samples / params.CHUNK_SAMPLES))

            chunk_predictions = []

            for i in range(num_chunks):
                start = i * params.CHUNK_SAMPLES
                end = start + params.CHUNK_SAMPLES
                chunk = audio_array[start:end]

                # ✅ Pad with silence (zeros) if shorter than 30s
                if len(chunk) < params.CHUNK_SAMPLES:
                    padding = np.zeros(params.CHUNK_SAMPLES - len(chunk), dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, padding])

                # Now chunk is guaranteed to be exactly 30s
                assert len(chunk) == params.CHUNK_SAMPLES, f"Chunk {i} has {len(chunk)} samples!"

                # Process with Whisper
                input_features = processor(
                    chunk,
                    sampling_rate=params.SAMPLING_RATE,
                    return_tensors="pt"
                ).input_features.to(params.torch_dtype).to(DEVICE)

                with torch.no_grad():
                    pred_ids = model.generate(input_features)[0]

                pred_str = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                pred_str = normalizer(pred_str)
                chunk_predictions.append(pred_str)

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
    average_wer_per_class[lab] = np.mean(all_wer_per_class[lab])
    average_cer_per_class[lab] = np.mean(all_cer_per_class[lab])

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
print("average_wer_per_class")
print(*wer_w_list, sep='\n')
print("average_cer_per_class")
print(*cer_w_list, sep='\n')