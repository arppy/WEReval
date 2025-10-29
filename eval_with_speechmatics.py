import asyncio
import argparse
import evaluate
import csv
from speechmatics.batch import AsyncClient, JobConfig, JobType, TranscriptionConfig
from pathlib import Path
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from utils import get_LaciControl_as_list, get_LaciDys_as_list, get_Bea_as_list, get_HunDys_as_list
import params


async def transcribe_file(client, file_path, lab, normalizer, metric_wer, metric_cer, ground_truth_text):
    normalized_reference = normalizer(ground_truth_text)
    """Transcribe one file and return result dict."""
    try:
        config = JobConfig(
            type=JobType.TRANSCRIPTION,
            transcription_config=TranscriptionConfig(
                language="hu",  # adjust if needed per dataset
                diarization="none"  # or "speaker" if you need speaker labels
            )
        )
        job = await client.submit_job(str(file_path), config=config)
        result = await client.wait_for_completion(job.id, timeout=600.0)  # 10 min timeout

        raw_transcript = result.transcript_text or ""
        normalized_transcript = normalizer(raw_transcript)


        wer = metric_wer.compute(predictions=[normalized_transcript], references=[normalized_reference])
        cer = metric_cer.compute(predictions=[normalized_transcript], references=[normalized_reference])
        word_n = len(normalized_transcript.split())
        char_n = len(normalized_transcript.replace(" ", ""))

        return {
            "file_path": str(file_path),
            "class_label": str(lab),
            "expected_text": normalized_reference,
            "transcription": normalized_transcript,
            "wer": str(wer),
            "cer": str(cer),
            "word_count": str(word_n),
            "char_count": str(char_n)
        }

    except asyncio.TimeoutError:
        return {
            "file_path": str(file_path),
            "class_label": str(lab),
            "expected_text": normalized_reference,
            "transcription": "[TIMEOUT]",
            "wer": "-1.0",
            "cer": "-1.0",
            "word_count": "0",
            "char_count": "0",
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": str(file_path),
            "class_label": str(lab),
            "expected_text": normalized_reference,
            "transcription": "[ERROR]",
            "wer": "-1.0",
            "cer": "-1.0",
            "word_count": "0",
            "char_count": "0",
        }

async def main(api_key: str, to_process, file_paths, texts, labels, output_file):
    # Build ground truth map again for this batch
    normalizer = BasicTextNormalizer()
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")
    file_to_info_local = {str(fp): (txt, lbl) for fp, txt, lbl in zip(file_paths, texts, labels)}

    results = []

    # Use semaphore to limit concurrent jobs (e.g., 5 at a time)
    semaphore = asyncio.Semaphore(5)

    async def process_with_limit(path):
        async with semaphore:
            text, lab = file_to_info_local[str(path)]
            result = await transcribe_file(client, path, lab, normalizer, metric_wer, metric_cer, text)
            results.append(result)
            # Append to CSV immediately (optional but safe)
            with open(output_file, mode='a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "file_path", "transcription", "normalized_transcription",
                    "reference", "normalized_reference", "wer", "cer", "status"
                ])
                if f.tell() == 0:  # write header if file is empty
                    writer.writeheader()
                writer.writerow(result)
            print(f"Completed: {path.name}")

    async with AsyncClient(api_key=api_key) as client:
        tasks = [process_with_limit(path) for path in to_process]
        await asyncio.gather(*tasks)

    print(f"Processed {len(results)} files. Results appended to {output_file}")

parser = argparse.ArgumentParser(description="Evaluation with Speechmatics.")
parser.add_argument("dataset", metavar="dataset", type=str, help="Name of dataset.")
parser.add_argument("api_key", metavar="api-key", type=str, help="Speechmatics API key")
parser.add_argument("wav_dir", metavar="wav-dir", type=Path, help="path to audio directory.")
parser.add_argument("output_file", metavar="output-file", type=Path, help="path to output file.")
args = parser.parse_args()

if args.dataset == params.LACICON:
    file_paths, texts, labels = get_LaciControl_as_list(args.wav_dir)
elif args.dataset == params.LACIDYS:
    file_paths, texts, labels = get_LaciDys_as_list(args.wav_dir)
elif args.dataset == params.BEA:
    file_paths, texts, labels = get_Bea_as_list(args.wav_dir)
elif args.dataset == params.HUNDYS:
    file_paths, texts, labels = get_HunDys_as_list(args.wav_dir)

# Open output CSV file
output_file = Path(args.output_file)
# ----------------------------
# STEP 1: Load existing results (if file exists)
# ----------------------------
existing_results = {}
if output_file.exists():
    with open(output_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_results[row["file_path"]] = row
    print(f"Loaded {len(existing_results)} existing results from {output_file}")
else:
    print("No existing results. Starting fresh.")
# ----------------------------
# STEP 2: Determine which files need processing
# ----------------------------
to_process = []
for i, fp in enumerate(file_paths):
    fp_str = str(fp)
    existing = existing_results.get(fp_str)
    if existing is None or existing["transcription"] == "TIMEOUT":
        to_process.append(fp)
print(f"Files to process/retry: {len(to_process)}")

# ----------------------------
# STEP 3: Process missing or failed files
# ----------------------------
if to_process:
    asyncio.run(main(args.api_key, to_process, to_process, file_paths, texts, labels, output_file))