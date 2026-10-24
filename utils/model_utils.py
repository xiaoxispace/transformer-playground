import copy
import torch
import json

from pathlib import Path

from models.transformer import Transformer
from utils.tokenizer_utils import load_tokenizer


MODEL_PARAMETERS = (
  "vocab_size",
  "block_size",
  "dropout",
  "dmodel",
  "num_of_encoder_layers", 
  "num_of_decoder_layers",
  "num_of_heads",
  "decoder_only",
)


def extract_model_config(config):
  model_config = {}
  for k, v in config.items():
    if k in MODEL_PARAMETERS:
      model_config[k] = v

  return model_config


def make_model_saver(config, tokenizer, run_name=None, save_dir="checkpoints", tokenizer_subpath="tokenizer"):
    def save_model(model, epoch, is_best_model=False):
      if model.decoder_only:
        filename = (
          f"_ctx{config['block_size']}"
          f"_d{config['dmodel']}"
          f"_L{config['num_of_decoder_layers']}"
          f"_H{config['num_of_heads']}"
          f"_lr{config['learning_rate']:.0e}"
          f"_dropout{config['dropout']:.0e}"
          f"_e{epoch}.pth"
        )
      else:
        filename = (
          f"_ctx{config['block_size']}"
          f"_d{config['dmodel']}"
          f"_dcdr{config['num_of_decoder_layers']}"
          f"_ecdr{config['num_of_encoder_layers']}"
          f"_H{config['num_of_heads']}"
          f"_lr{config['learning_rate']:.0e}"
          f"_dropout{config['dropout']:.0e}"
          f"_e{epoch}.pth"
        )

      if is_best_model:
        filename = "best_model" + filename
      else:
        filename = "model" + filename

      target_dir = Path(save_dir)
      if run_name:
        run_dir = target_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / filename
        tokenizer_path = run_dir / tokenizer_subpath
        model_config_path = run_dir / 'config.json'
      else:
        model_path = target_dir / filename
        tokenizer_path = target_dir / tokenizer_subpath
        model_config_path = target_dir / 'config.json'

      if not model_config_path.is_file():
        with open(model_config_path, "w", encoding="utf-8") as f:
          json.dump(config, f, indent=2)

      tokenizer.save(tokenizer_path)
      torch.save(model.state_dict(), model_path)

      return model_path, tokenizer_path

    return save_model


def load_model(model_path, config='config.json', tokenizer_path="tokenizer", map_location="cuda"):
  model_dir = Path(model_path).parent
  tokenizer_path = model_dir / tokenizer_path
  config_path = model_dir / config

  with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

  tokenizer = load_tokenizer(tokenizer_path)

  model = Transformer(**extract_model_config(config))
  model.load_state_dict(torch.load(model_path, map_location=map_location))

  return model, tokenizer, config