import os
from pathlib import Path
import json
import numpy as np
import torch
import evaluate
import argparse

from transformers import (WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer

from accelerate import Accelerator

import params
from utils import load_dataset_for_ASR, DataCollatorSpeechSeq2SeqWithPadding

accelerator = Accelerator()
DEVICE = accelerator.device
CPU = torch.device('cpu')

parser = argparse.ArgumentParser(description="Evaluation with ASR.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
args = parser.parse_args()
model = WhisperForConditionalGeneration.from_pretrained(params.whisper_arch, torch_dtype=params.torch_dtype)
model = model.to(DEVICE)
model.config.forced_decoder_ids = None
if params.lang == "en" :
    processor = WhisperProcessor.from_pretrained(params.whisper_arch, language="english", task="transcribe", torch_dtype=params.torch_dtype)
    model.generation_config.language = "english"
    mapping_path = os.path.join(os.path.dirname("imports/"), "english.json")
    english_spelling_mapping = json.load(open(mapping_path))
    normalizer = EnglishTextNormalizer(english_spelling_mapping)
else :
    processor = WhisperProcessor.from_pretrained(params.whisper_arch, language="hungarian", task="transcribe", torch_dtype=params.torch_dtype)
    model.generation_config.language = "hungarian"
    normalizer = BasicTextNormalizer()

model.generation_config.task = "transcribe"
#augmentor = SpeedAugmentation(params.SAMPLING_RATE, params.speed_factor)

fn_kwargs = {"feature_extractor":  processor.feature_extractor,
             "tokenizer": processor.tokenizer,
             "augmentor": None}


#dataset_train = load_UASpeech_dataset(params.TRAIN_SPEAKERS, fn_kwargs)
#dataset_test = load_UASpeech_dataset(params.TEST_SPEAKERS, fn_kwargs)
dataset_testds = load_dataset_for_ASR(params.dataset, params.TEST_DYSARTHRIC_SPEAKERS, args.wav_dir, fn_kwargs, True)
#test_loader = torch.utils.data.DataLoader(dataset, batch_size=params.per_device_train_batch_size)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

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
for i in range(params.label_count[params.dataset]) :
    all_wer_per_class[i] = []
    all_wN_per_class[i] = []
    all_cer_per_class[i] = []
    all_cN_per_class[i] = []
    all_transcriptions_str_per_class[i] = ""
    all_expects_str_per_class[i] = ""
    average_wer_per_class.append(-1)
    average_cer_per_class.append(-1)
i = 0
j = 0
with open('transcriptions_'+params.dataset+'_testdys.txt', 'w') as file, open('expected_texts_'+params.dataset+'.txt', 'w') as expected_file:
    for test_record in dataset_testds:
        collated_test_record = data_collator([test_record])
        input_features = collated_test_record["input_features"].to(params.torch_dtype).to(DEVICE)
        pred = model.generate(input_features)

        pred_ids = pred[0]
        label_ids = collated_test_record['labels'].to(DEVICE)

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_ids = label_ids[0]

        # we do not want to group tokens when computing the metrics
        pred_str = normalizer(processor.tokenizer.decode(pred_ids, skip_special_tokens=True))
        label_str = normalizer(processor.tokenizer.decode(label_ids, skip_special_tokens=True))

        lab = test_record['severity']

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
        i += 1

        #print(average_wer_per_class, average_cer_per_class)
        # Write each transcription to the file
        file.write(str(j) + ':' + pred_str + '\n')
        expected_file.write(str(j) + ':' + label_str + '\n')
        j += 1

wer_a_list = []
wer_w_list = []
cer_a_list = []
cer_w_list = []
for lab in range(params.label_count[params.dataset]) :
    if all_transcriptions_str_per_class[lab] != "" :
        wer_a_list.append(metric_wer.compute(predictions=[all_transcriptions_str_per_class[lab]], references=[all_expects_str_per_class[lab]]))
        cer_a_list.append(metric_cer.compute(predictions=[all_transcriptions_str_per_class[lab]], references=[all_expects_str_per_class[lab]]))
        np_all_wer_per_lab = np.array(all_wer_per_class[lab])
        np_all_wN_per_lab = np.array(all_wN_per_class[lab])
        np_all_cer_per_lab = np.array(all_cer_per_class[lab])
        np_all_cN_per_lab = np.array(all_cN_per_class[lab])
        wer_w_list.appen(np.average(np_all_wer_per_lab, weights=np_all_wN_per_lab))
        cer_w_list(np.average(np_all_cer_per_lab, weights=np_all_cN_per_lab))
    else:
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