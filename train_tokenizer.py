"""
This file takes output.txt, the concatenated play corpus, and then tokenize it. 
The tokenizer groups consecutive characters of the same type (alpha, digit, or else).
Executing this file produces 
  1. tokenizer mapping in json, which is a 1-to-1 mapping from string to integer and
  2. token index of the corpus stored in pytorch tensor of type torch.int32
  3. a json string-to-integer map that denotes each token's number of occurrences. 
"""
import argparse
import logging
import random
import matplotlib.pyplot as plt

from pathlib import Path

from data_config import TOKENIZER_DATA_PATH
from hyperparams import config
from utils.tokenizer_utils import create_tokenizer, decide_tokenizer_data_path, extract_tokenizer_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tokenizer")


def load_corpus(corpus_path):
  with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = f.read()
  return corpus


def display_token(token):
  """Normalize displayed tokens on frequency figures"""
  if token == " ":
    return "<space>"
  elif token == "\n":
    return "<newline>"
  elif token == "\t":
    return "<tab>"
  elif token == "\r":
    return "<return>"
  else:
    return token


def plot_token_counts(token_counts, num_random=15):
  # -------------------------
  # 1. Top 15 most frequent words
  # -------------------------
  tokens = sorted(
    token_counts.items(),
    key=lambda x: x[1],
    reverse=True
  )[:15]
  counts = [count for _, count in tokens]
  displayed_tokens = [display_token(token) for token, _ in tokens]

  plt.figure(figsize=(12, 5))
  plt.bar(displayed_tokens, counts)
  plt.xlabel("Token")
  plt.ylabel("Count")
  plt.title("Top 15 Most Frequent Tokens")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig("figs/top_15_tokens.png", dpi=300)
  plt.close()

  # -------------------------
  # 2. Randomly sampled tokens
  # -------------------------
  num_random = min(num_random, len(token_counts))

  random.seed(6)
  sampled_tokens = random.sample(
    [token for token in token_counts.keys()],
    num_random
  )

  sampled_counts = [
    token_counts[token]
    for token in sampled_tokens
  ]
  sampled_tokens = [display_token(t) for t in sampled_tokens] # normalize for display

  plt.figure(figsize=(12, 5))
  plt.bar(sampled_tokens, sampled_counts)
  plt.xlabel("Token")
  plt.ylabel("Count")
  plt.title(f"Random Sample of {num_random} Tokens")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig("figs/random_tokens.png", dpi=300)
  plt.close()


dataset_to_tokenizer_suffix = {
  "shakespeare": "default",
  "emed": "emed"
}

def main():
  parser = argparse.ArgumentParser(
                  prog='tokenizer-training',
                  description='train a tokenizer for transformer')
  parser.add_argument('--dataset-paths', nargs='+', type=Path)
  args = parser.parse_args()
  dataset_paths = args.dataset_paths
  corpus = ""
  for p in dataset_paths:
    corpus += "\n" + load_corpus(p)
  corpus = corpus[1:]

  dataset_identifier = "-".join([dataset_to_tokenizer_suffix[p.parent.name] for p in dataset_paths])

  tokenizer_config = extract_tokenizer_config(config)
  tokenizer = create_tokenizer(tokenizer_config)
  tokenizer.train(corpus)
  tokenizer_data_path = decide_tokenizer_data_path(
    tokenizer_config, prefix=TOKENIZER_DATA_PATH, dataset_identifier=dataset_identifier
  )
  tokenizer.save(tokenizer_data_path)

  # token_ids = tokenizer.encode(corpus)
  # counter = Counter(token_ids)
  # token_counts = { tokenizer.decode([i]): counter[i] for i in counter }
  # plot_token_counts(token_counts)


if __name__ == "__main__":
  main()