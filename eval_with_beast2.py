from playwright.sync_api import sync_playwright
from pathlib import Path
import argparse
import evaluate

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from utils import get_LaciControl_as_list, get_LaciDys_as_list, get_Bea_as_list, get_HunDys_as_list
import params

parser = argparse.ArgumentParser(description="Evaluation with BEAST2 ASR.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
args = parser.parse_args()

if args.dataset == params.LACICON:
    file_paths, texts, labels = get_LaciControl_as_list(args.wav_dir)
elif args.dataset == params.LACIDYS:
    file_paths, texts, labels = get_LaciDys_as_list(args.wav_dir)
elif args.dataset == params.BEA:
    file_paths, texts, labels = get_Bea_as_list(args.wav_dir)
elif args.dataset == params.HUNDYS:
    file_paths, texts, labels = get_HunDys_as_list(args.wav_dir)

normalizer = BasicTextNormalizer()

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
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://phon.nytud.hu/beast2")
    for audio_file in file_paths:
        # 1. Clear if needed (optional)
        page.click("#component-4")

        # 2. Upload
        page.set_input_files("#component-2 input[type='file']", audio_file)

        # 3. Wait for waveform/duration → upload complete
        page.wait_for_selector("#component-2 .waveform-container", timeout=30000)

        # 4. Now click Run — safe!
        page.click("#component-5")

        # Wait until the textarea has non-empty value
        page.wait_for_function("""
            () => {
                const textarea = document.querySelector('#component-10 textarea');
                return textarea && textarea.value.trim() !== '' && !textarea.value.match(/^\\d+\\.\\d+s$/);
            }
        """, timeout=60000)

        # Extract the actual value from the textarea
        result = page.eval_on_selector("#component-10 textarea", "el => el.value")
        pred_str = normalizer(result.strip())
        print(audio_file,result.strip())
    browser.close()