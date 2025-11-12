import re
import csv
import argparse
import evaluate
import params
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description="Evaluation with BEAST2 ASR csv file for separate wav file concat.")
parser.add_argument("csv_file", metavar="csv-file", type=Path, help="path to input csv file.")
args = parser.parse_args()

transcriptions = {}
with open(args.csv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_path = row['file_path']
        # Convert numeric fields to appropriate types (optional but recommended)
        transcriptions[file_path] = {
            'class_label': int(row['class_label']),
            'expected_text': row['expected_text'],
            'transcription': row['transcription'],
            'wer': float(row['wer']) if row['wer'] else None,
            'cer': float(row['cer']) if row['cer'] else None,
            'word_count': int(row['word_count']) if row['word_count'] else None,
            'char_count': int(row['char_count']) if row['char_count'] else None
        }

# Group files for concatenation
file_groups = {}
ungrouped_records = []

# Process all records and separate into groups and ungrouped
for file_path_name in transcriptions:
    file_path = Path(file_path_name)
    name = file_path.stem
    # Match pattern like M001_09_1, CF014_09_01, etc. (has number at the end after underscores)
    matcher = re.match(r'^([A-Z]+\d+_\d+)_(\d+)$', name)
    if matcher:
        # Additional check: make sure it's not a complex filename with descriptive text
        prefix = matcher.group(1)  # e.g., "M001_09", "CF014_09"
        underscore_count = name.count('_')
        if prefix not in file_groups:
            file_groups[prefix] = []
        file_groups[prefix].append(file_path_name)
    else:
        ungrouped_records.append(file_path_name)

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
# Process grouped files
for prefix, group_file_paths in file_groups.items():
    # Sort by sequence number
    group_file_paths.sort(key=lambda x: int(re.search(r'_(\d+)$', Path(x).stem).group(1)))
    concatenated_pred_str = ""
    lab = transcriptions[group_file_paths[0]]['class_label']  # Assume same severity for all in group
    label_str = transcriptions[group_file_paths[0]]['expected_text']
    path = Path(group_file_paths[0])
    stem = path.stem
    base_name = re.sub(r'_\d+$', '', stem)
    modified_path = path.parent / f"{base_name}.wav"
    for file_path in group_file_paths:
        pred_str = transcriptions[file_path]['transcription']
        concatenated_pred_str += " " + pred_str
    pred_str = concatenated_pred_str.strip()

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

for file_path in ungrouped_records:
    lab = transcriptions[file_path]['class_label']
    pred_str = transcriptions[file_path]['transcription']
    label_str = transcriptions[file_path]['expected_text']

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
