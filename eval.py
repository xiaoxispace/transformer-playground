"""This file reports basic NLP metrics including perplexity."""
import argparse
import torch

from dataset import prepare_dataset_loader
from pathlib import Path

from hyperparams import config
from utils.eval_utils import perplexity
from utils.model_utils import load_model


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)

  train_end = 0.7  # same as train.py
  val_end = 0.9  # same as train.py
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model, tokenizer, model_config = load_model(model_path, map_location=device)
  model.to(device)

  print("Load dataset to compute metrics.")
  dataset_path = Path("data/output.txt")
  with open(dataset_path, 'r', encoding='utf-8') as file:
    raw_tokens = file.read()
  token_ids = tokenizer.encode(raw_tokens)
  token_sequence = torch.tensor(token_ids).to(torch.long)
  train_loader, val_loader, test_loader = prepare_dataset_loader(token_sequence, model_config["block_size"], config["batch_size"], train_end, val_end)
  print("Load done.\n")

  per = perplexity(model, test_loader, device)
  print(f"Perlexity: {per}")