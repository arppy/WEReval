import re
import argparse
import evaluate
import params
from pathlib import Path
import numpy as np
import json

parser = argparse.ArgumentParser(description="Evaluation with FastConformer_HU ASR json file for separate wav file concat.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("json_files_path", metavar="json-file", type=Path, help="path to input json files.")
args = parser.parse_args()

text_directory_path = Path(args.wav_dir).parent / "text"
transcriptions = []
for json_file in Path(args.json_files_path).iterdir():
    if json_file.suffix == '.json':
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            transcription = data['transcription']
            transcriptions.append(transcription)
        except KeyError:
            print(f"No transcription field in {json_file.name}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {json_file.name}")