import os
import random
import zipfile
import pickle
import gc
from tqdm import tqdm

from dataset import process_sgf_to_samples_from_text
from utils import BOARD_SIZE, HISTORY_LENGTH

# ===== フォルダ設定 =====
SGF_INPUT_DIR   = r"D:\igo\simple2_sgf\temp"
OUTPUT_DIR      = r"D:\igo\test\python\bitpack\zip"
SGF_PER_BATCH   = 1000
AUGMENT_ALL     = 2
RANDOM_SEED     = 42

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    all_sgfs = [f for f in os.listdir(SGF_INPUT_DIR) if f.endswith('.sgf')]
    if not all_sgfs:
        tqdm.write(f"[ERROR] No SGF files found in {SGF_INPUT_DIR}")
        return

    tqdm.write(f"[INFO] Total SGF files found: {len(all_sgfs)}")
    random.shuffle(all_sgfs)

    batches = [all_sgfs[i:i + SGF_PER_BATCH] for i in range(0, len(all_sgfs), SGF_PER_BATCH)]

    for batch_idx, sgf_list in enumerate(tqdm(batches, desc="Generating SGF zip batches", ncols=100)):
        batch_samples = []

        for sgf_name in tqdm(sgf_list, desc=f" Batch {batch_idx:04d}", leave=False, ncols=80):
            sgf_path = os.path.join(SGF_INPUT_DIR, sgf_name)
            try:
                with open(sgf_path, "r", encoding="utf-8") as f:
                    sgf_text = f.read()
                samples = process_sgf_to_samples_from_text(
                    sgf_text,
                    board_size=BOARD_SIZE,
                    history_length=HISTORY_LENGTH,
                    augment_all=AUGMENT_ALL
                )
                batch_samples.extend(samples)
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to process {sgf_name}: {e}")

        zip_path = os.path.join(OUTPUT_DIR, f"sgf_batch_{batch_idx:04d}.zip")
        try:
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open("samples.pkl", "w") as f:
                    pickle.dump(batch_samples, f)
            tqdm.write(f"[INFO] Saved: {zip_path} ({len(batch_samples)} samples)")
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to write zip: {zip_path}: {e}")

        del batch_samples
        gc.collect()

if __name__ == "__main__":
    main()
