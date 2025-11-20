import torchaudio
from pathlib import Path
import argparse
import params
from utils import SzegedDys_label_to_idx

parser = argparse.ArgumentParser(description="Evaluation with ASR.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
args = parser.parse_args()

wav_files = list(Path(args.wav_dir).glob("*.wav"))

all_length_per_class = {}
for i in range(params.label_count[args.dataset]) :
    all_length_per_class[i] = 0
for i in range(0, len(wav_files)):
    wav_file = wav_files[i]
    wav_file_name_parts = wav_file.stem.split("_")
    label = 0
    if args.dataset == params.HUNDYS :
        label = int(wav_file_name_parts[1]) - 1
    elif args.dataset == params.SZEGEDYS :
        label = SzegedDys_label_to_idx[wav_file_name_parts[0]]
    try:
        waveform, sample_rate = torchaudio.load(str(wav_file))
        duration = waveform.shape[1] / sample_rate
        all_length_per_class[label] += duration
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
for label in range(params.label_count[args.dataset]) :
    if all_length_per_class[label] == 0 :
        print(-1)
    else:
        print(all_length_per_class[label])