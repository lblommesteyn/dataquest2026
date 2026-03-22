"""
Download the PhysioNet 2019 Sepsis Challenge data from Kaggle.

Dataset: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

Setup:
  1. Go to https://www.kaggle.com/settings/account → Create New Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. Run: python download_data.py

The data will land in data/raw/ as training_setA.csv and training_setB.csv.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
KAGGLE_DATASET = "salikhussaini49/prediction-of-sepsis"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (RAW_DIR / "training_setA.csv").exists():
        print("training_setA.csv already exists, skipping download.")
        print("Delete data/raw/training_setA.csv to re-download.")
        return

    print(f"Downloading {KAGGLE_DATASET} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR), "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\nDownload failed. Make sure your kaggle.json is set up:")
        print("  1. Go to https://www.kaggle.com/settings/account → Create New Token")
        print("  2. Save kaggle.json to ~/.kaggle/kaggle.json")
        sys.exit(1)

    # List what arrived
    files = list(RAW_DIR.iterdir())
    print(f"\nDownloaded files in {RAW_DIR}:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    print("\nDone! Now run: python train.py")


if __name__ == "__main__":
    main()
