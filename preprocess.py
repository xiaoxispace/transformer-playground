import argparse

from data_config import DATASET_TO_PREPROCESS_DIRS

from utils.preprocess_utils import get_preprocessor


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                  prog='playground-preprocessing',
                  description='preprocess transformer datasets')
  parser.add_argument('-d', '--dataset', choices=["shakespeare", "emed"], type=str.lower)
  args = parser.parse_args()
  dataset_name = args.dataset

  prp = get_preprocessor(dataset_name)
  text_input_path, text_output_path = DATASET_TO_PREPROCESS_DIRS[dataset_name]
  prp.preprocess(text_input_path, text_output_path)