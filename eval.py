"""This file reports basic NLP metrics including perplexity."""
import argparse
import torch

from utils.dataset_utils import prepare_dataset_loader
from pathlib import Path

from hyperparams import config
from utils.eval_utils import perplexity
from utils.model_utils import load_model


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)
  argparser.add_argument('--dataset-paths', nargs='+', type=str)

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)
  dataset_paths = parser.dataset_paths

  train_end = 0.7  # same as train.py
  val_end = 0.9  # same as train.py
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model, tokenizer, model_config = load_model(model_path, map_location=device)
  model.to(device)

  print("Load dataset to compute metrics.")
  _, _, test_loader = prepare_dataset_loader(dataset_paths, tokenizer, config['train_end'], config['val_end'], model_config["block_size"], config["batch_size"])
  print("Load done.\n")

  per = perplexity(model, test_loader, device)
  print(f"Perlexity: {per}")