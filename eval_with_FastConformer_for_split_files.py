import argparse
import evaluate
import params
from pathlib import Path
import numpy as np
import json
import re

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from utils import get_LaciControl_as_list, get_LaciDys_as_list, get_Bea_as_list, get_HunDys_as_list, \
    get_SzegedDys_as_list

parser = argparse.ArgumentParser(description="Evaluation with FastConformer_HU ASR json file for separate wav file concat.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("json_files_path", metavar="json-file", type=Path, help="path to input json files.")
args = parser.parse_args()

file_paths = []
texts = []
labels = []
if args.dataset == params.LACICON:
    file_paths, texts, labels = get_LaciControl_as_list(args.wav_dir)
elif args.dataset == params.LACIDYS:
    file_paths, texts, labels = get_LaciDys_as_list(args.wav_dir)
elif args.dataset == params.BEA:
    file_paths, texts, labels = get_Bea_as_list(args.wav_dir)
elif args.dataset == params.HUNDYS:
    file_paths, texts, labels = get_HunDys_as_list(args.wav_dir)
elif args.dataset == params.SZEGEDYS:
    file_paths, texts, labels = get_SzegedDys_as_list(args.wav_dir)

normalizer = BasicTextNormalizer()
metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

file_to_info = {str(fp): (txt, lbl) for fp, txt, lbl in zip(file_paths, texts, labels)}
text_directory_path = Path(args.wav_dir).parent / "text"

# Group files for concatenation
file_groups = {}
ungrouped_records = []
# Process all records and separate into groups and ungrouped
for file_path_name in Path(args.json_files_path).iterdir():
    file_path = Path(file_path_name)
    name = Path(file_path.stem).stem
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

all_wer_per_class = {}
all_wN_per_class = {}
all_cer_per_class = {}
all_cN_per_class = {}
average_wer_per_class = []
average_cer_per_class = []
for i in range(params.label_count[args.dataset]):
    all_wer_per_class[i] = []
    all_wN_per_class[i] = []
    all_cer_per_class[i] = []
    all_cN_per_class[i] = []
    average_wer_per_class.append(-1)
    average_cer_per_class.append(-1)
for json_file_name in ungrouped_records:
    json_file = Path(json_file_name)
    if json_file.suffix == '.json':
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            pred_str = normalizer(data['transcription'])
            json_wav_path = str(Path(args.wav_dir) / json_file.stem)
            if json_wav_path in file_to_info:
                label_str_orig, lab = file_to_info[json_wav_path]
                label_str = normalizer(label_str_orig)
                word_N = len(label_str.split())
                char_N = len(label_str.replace(" ", ""))
                wer = metric_wer.compute(predictions=[pred_str], references=[label_str])
                cer = metric_cer.compute(predictions=[pred_str], references=[label_str])
                all_wer_per_class[lab].extend([wer])  # Add the current batch's WERs to the list
                all_wN_per_class[lab].extend([word_N])  # Add the current batch's WERs to the list
                all_cer_per_class[lab].extend([cer])  # Add the current batch's CERs to the list
                all_cN_per_class[lab].extend([char_N])  # Add the current batch's CERs to the list
        except KeyError:
            print(f"No transcription field in {json_file.name}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {json_file.name}")
for prefix, group_file_paths in file_groups.items():
    # Sort by sequence number
    group_file_paths.sort(key=lambda x: int(re.search(r'_(\d+)$', Path(Path(x).stem).stem).group(1)))
    concatenated_pred_str = ""
    json_wav_path = str(Path(args.wav_dir) / (prefix + ".wav"))
    if json_wav_path in file_to_info:
        label_str_orig, lab = file_to_info[json_wav_path]
        label_str = normalizer(label_str_orig)
        word_N = len(label_str.split())
        char_N = len(label_str.replace(" ", ""))
    else:
        continue
    for json_file_name in group_file_paths:
        json_file = Path(json_file_name)
        if json_file.suffix == '.json':
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                pred_str = normalizer(data['transcription'])
                concatenated_pred_str += " " + pred_str
            except KeyError:
                print(f"No transcription field in {json_file.name}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in {json_file.name}")
    pred_str = concatenated_pred_str.strip()
    wer = metric_wer.compute(predictions=[pred_str], references=[label_str])
    cer = metric_cer.compute(predictions=[pred_str], references=[label_str])
    all_wer_per_class[lab].extend([wer])  # Add the current batch's WERs to the list
    all_wN_per_class[lab].extend([word_N])  # Add the current batch's WERs to the list
    all_cer_per_class[lab].extend([cer])  # Add the current batch's CERs to the list
    all_cN_per_class[lab].extend([char_N])  # Add the current batch's CERs to the list

wer_w_list = []
cer_w_list = []
for lab in range(params.label_count[args.dataset]) :
    if len(all_wN_per_class[lab])>0 :
        np_all_wer_per_lab = np.array(all_wer_per_class[lab])
        np_all_wN_per_lab = np.array(all_wN_per_class[lab])
        np_all_cer_per_lab = np.array(all_cer_per_class[lab])
        np_all_cN_per_lab = np.array(all_cN_per_class[lab])
        # Check if weights sum to zero
        if np_all_wN_per_lab.sum() == 0:
            wer_w_list.append(-1.0)  # or np.nan, or skip
        else:
            wer_w_list.append(np.average(np_all_wer_per_lab, weights=np_all_wN_per_lab))
        if np_all_cN_per_lab.sum() == 0:
            cer_w_list.append(-1.0)
        else:
            cer_w_list.append(np.average(np_all_cer_per_lab, weights=np_all_cN_per_lab))
    else :
        wer_w_list.append(-1.0)
        cer_w_list.append(-1.0)
print("average_wer_per_class")
print(*wer_w_list, sep='\n')
print("average_cer_per_class")
print(*cer_w_list, sep='\n')