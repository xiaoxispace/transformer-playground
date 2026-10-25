import argparse

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