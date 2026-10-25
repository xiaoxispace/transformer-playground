import json
import torch

from collections import Counter
from pathlib import Path
from tqdm import tqdm

from tokenizers.tokenizer_abc import TokenizerABC


class CharTokenizer(TokenizerABC):
  def __init__(self):
    super(CharTokenizer).__init__()
    self.vocab_to_ind = {}
    self.ind_to_vocab = {}
    self.config = {"tokenizer": "char"}

  def train(self, text, **kwargs):
    raw_tokens = self._tokenize_play(text)
    self.vocab_to_ind, self.token_counts = self._build_vocabulary(raw_tokens)
    counts = torch.tensor(list(self.token_counts.values()), dtype=torch.int32)
    print("Summary statistics:")
    print(f"Token number: {len(self.token_counts)}")
    print(f"Text raw token count: {len(raw_tokens)}")
    print("Tokens appearing once:", (counts == 1).sum())
    print("Tokens appearing <= 5:", (counts <= 5).sum())
    print("Tokens appearing <= 10:", (counts <= 10).sum())
    print("Tokens appearing >= 100:", (counts >= 100).sum())
    print("Tokens appearing >= 1000:", (counts >= 1000).sum())

  def encode(self, text):
    return [self.vocab_to_ind[s] for s in text]

  def decode(self, token_ids):
    if not self.ind_to_vocab:
      self.ind_to_vocab = {v :k for k, v in self.vocab_to_ind.items()}
    return ''.join([self.ind_to_vocab[i] for i in token_ids])

  def save(self, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab_to_ind_path = path / "vocab_to_ind.json"
    with open(vocab_to_ind_path, "w", encoding="utf-8") as file:
      json.dump(self.vocab_to_ind, file, indent=4)
    config_path = path / "config.json"
    with open(config_path, "w", encoding="utf-8") as file:
      json.dump(self.config, file, indent=4)

  @classmethod
  def load(cls, path):
    path = Path(path)
    vocab_to_ind_path = path / "vocab_to_ind.json"
    with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
      vocab_to_ind = json.load(file)
    config_path = path / "config.json"
    with open(config_path, "r", encoding="utf-8") as file:
      config = json.load(file)

    t = cls()
    t.vocab_to_ind = vocab_to_ind
    t.config = config
    return t

  @property
  def vocab_size(self):
    return len(self.vocab_to_ind)

  def _tokenize_play(self, text):
    """Tokenize text at character level."""
    raw_tokens = []
    for i in tqdm(
      range(len(text)),
      desc=f"Collect chars from text of size {len(text)}"
    ):
      c = text[i]
      raw_tokens.append(c)
    return raw_tokens

  def _build_vocabulary(self, raw_tokens):
    """Build token-to-index and token-count dictionaries."""
    token_counts = Counter(raw_tokens)

    # Sort for deterministic vocabulary construction.
    vocab = sorted(token_counts.keys(), reverse=True) 
      # make it clear that sort on the reverse of token's lexical order

    vocab_to_ind = {
      token: i
      for i, token in enumerate(vocab)
    }

    return vocab_to_ind, dict(token_counts)




