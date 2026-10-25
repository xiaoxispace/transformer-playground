"""
This file contains the general configurations for dataset paths, tensor paths, etc. under
the "data/" directory.
"""
from pathlib import Path

DATASETS = ("shakespeare", "emed")


DATASETS_DIR = Path("data/datasets")

SHAKESPEARE_PLAY_DIR = DATASETS_DIR / "shakespeare" / "shakespeares-works_TXT_FolgerShakespeare"
SHAKESPEARE_OUTPUT_PATH = DATASETS_DIR / "shakespeare" / "output.txt"

EMED_DATASETS_DIR = DATASETS_DIR / "emed" / "plays"
EMED_OUTPUT_PATH = DATASETS_DIR / "emed" / "output.txt"

DATASET_TO_PREPROCESS_DIRS = {
  "shakespeare": [SHAKESPEARE_PLAY_DIR, SHAKESPEARE_OUTPUT_PATH],
  "emed": [EMED_DATASETS_DIR, EMED_OUTPUT_PATH]
}

TOKENIZER_DATA_PATH = Path("data/tokenizers")