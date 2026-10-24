import argparse
import json
import torch

from pathlib import Path

from utils.eval_utils import generate
from utils.model_utils import load_model


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)
  argparser.add_argument('-l', '--length', type=int, default=1000)
  argparser.add_argument('-s', '--start', type=str, default="to be")

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)
  length = parser.length
  start = parser.start
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model, tokenizer, config = load_model(model_path, map_location=device)
  model.to(device)
  model.eval()
  config_display = {k: str(v) for k, v in config.items()}
  print("Model config: ", json.dumps(config_display, indent=2))

  input_token_ids = tokenizer.encode(start)

  print("|| Begin generating contents: ||")
  output_token_ids = generate(model, input_token_ids, device=device)
  print(tokenizer.decode(output_token_ids))
