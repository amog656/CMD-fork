"""Helper script to download GCPG weights (>100MB) from Zenodo into the correct local directory."""

import urllib.request
from pathlib import Path
import sys

DEST = Path("cmdgen_repo/GCPG/checkpoints/model_checkpoints/checkpoints/final_model.pth")
ZENODO_URL = "https://zenodo.org/records/13841142/files/final_model.pth?download=1"

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading final_model.pth: {percent:.1f}% ({mb:.1f} / {total_mb:.1f} MB)")
        sys.stdout.flush()

def main():
    if DEST.is_file() and DEST.stat().st_size > 100 * 1024 * 1024:
        print(f"[OK] Weights file already exists at {DEST} ({DEST.stat().st_size / (1024*1024):.1f} MB).")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GCPG weights from Zenodo to {DEST}...")
    try:
        urllib.request.urlretrieve(ZENODO_URL, str(DEST), reporthook=download_progress)
        print("\n[OK] Weights successfully downloaded!")
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        print("Please download manually from https://zenodo.org/records/13841142 and place final_model.pth in:")
        print(f"  {DEST.resolve()}")

if __name__ == "__main__":
    main()
