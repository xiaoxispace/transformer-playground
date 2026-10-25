import json

from pathlib import Path

from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer


def decide_tokenizer_data_path(tokenizer_config, prefix, dataset_identifier='default'):
  tokenizer = tokenizer_config["tokenizer"]
  prefix = Path(prefix)
  if tokenizer == "char":
    return prefix / (tokenizer + "_" + dataset_identifier)
  elif tokenizer == "bpe":
    thresh = tokenizer_config["threshold"]
    return prefix / (tokenizer + "_" + dataset_identifier + "_" + f"threshold{thresh}")


def extract_tokenizer_config(config):
  """Convert general training interface config into tokenizer config"""
  t_name = config["tokenizer"]
  t_config = {
    "tokenizer": config["tokenizer"]
  }

  if t_name == 'bpe':
    t_config["threshold"] = config["bpe_vocab_threshold"]

  return t_config


def create_tokenizer(config):
  tokenizer_name = config["tokenizer"]
  if tokenizer_name == "char":
    return CharTokenizer()
  elif tokenizer_name == "bpe":
    return BPETokenizer(threshold=config["threshold"])
  raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def load_tokenizer(tokenizer_path):
  tokenizer_path = Path(tokenizer_path)
  tokenizer_config_path = tokenizer_path / "config.json"
  with open(tokenizer_config_path, "r", encoding="utf-8") as file:
    tokenizer_config = json.load(file)
  tokenizer_name = tokenizer_config["tokenizer"]

  if tokenizer_name == "char":
    return CharTokenizer.load(tokenizer_path)
  elif tokenizer_name == "bpe":
    return BPETokenizer.load(tokenizer_path)
  raise ValueError(f"Unknown tokenizer: {tokenizer_name}")