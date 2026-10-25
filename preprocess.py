"""
Running this preprocessing.py as main will do the following:

1. Reads all the plays under data/shakespeares-works_TXT_FolgerShakespeare
2. Strip all the comments before "Characters in the Play" (if any)
3. Remove all the lines of "=" separator in each play
4. Concatenate all the players into one single txt file

Note: The headers of plays without "Characters in Play" are removed manually.
"""
import argparse
import logging

from data_config import SHAKESPEARE_PLAY_DIR, SHAKESPEARE_OUTPUT_PATH

from utils.preprocess_utils import get_preprocessor


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                  prog='playground-preprocessing',
                  description='preprocess transformer datasets')
  parser.add_argument('--dataset', choices=["shakespeare"], type=str.lower)

  args = parser.parse_args()
  dataset_name = args.dataset

  prp = get_preprocessor(dataset_name)
  prp.preprocess(SHAKESPEARE_PLAY_DIR, SHAKESPEARE_OUTPUT_PATH)