from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

for directory in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

