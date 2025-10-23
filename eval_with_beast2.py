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
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://phon.nytud.hu/beast2")
    for i, audio_file in enumerate(file_paths):
        label_str = normalizer(texts[i])
        lab = labels[i]

        pred_str = None
        success = False

        for attempt in range(2):  # Try up to 2 times
            try:
                # 1. Clear if needed (optional)
                page.click("#component-4")
                # 2. Upload
                page.set_input_files("#component-2 input[type='file']", audio_file)
                # 3. Wait for waveform/duration → upload complete
                page.wait_for_selector("#component-2 .waveform-container", timeout=600000)
                # 4. Now click Run — safe!
                page.click("#component-5")
                # Wait until the textarea has non-empty value
                page.wait_for_function("""
                    () => {
                        const textarea = document.querySelector('#component-10 textarea');
                        return textarea && textarea.value.trim() !== '' && !textarea.value.match(/^\\d+\\.\\d+s$/);
                    }
                """, timeout=600000)
                # Extract the actual value from the textarea
                result = page.eval_on_selector("#component-10 textarea", "el => el.value")
                pred_str = normalizer(result.strip())
                success = True
                break  # Exit retry loop on success

            except PlaywrightTimeoutError:
                print(f"Timeout on {audio_file}, attempt {attempt + 1}/2")
                if attempt == 0:
                    # First timeout: retry
                    continue
                else:
                    # Second timeout: give up
                    print(f"Skipping {audio_file} after 2 timeouts.")
                    break

        if not success or pred_str is None:
            # Skip WER/CER for this file
            continue

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
    browser.close()

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