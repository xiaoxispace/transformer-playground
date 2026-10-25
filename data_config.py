"""
This file contains the general configurations for dataset paths, tensor paths, etc. under
the "data/" directory.
"""
from pathlib import Path

DATASETS = ("shakespeare")


DATASETS_DIR = Path("data/datasets")

SHAKESPEARE_PLAY_DIR = DATASETS_DIR / "shakespeare" / "shakespeares-works_TXT_FolgerShakespeare"
SHAKESPEARE_OUTPUT_PATH = DATASETS_DIR / "shakespeare" / "output.txt"

TOKENIZER_DATA_PATH = Path("data/tokenizers")