import base64
import json
import logging
import re

from collections import Counter
from pathlib import Path
from tqdm import tqdm

from tokenizers.tokenizer_abc import TokenizerABC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bpe-tokenizer")


def _base64_encode(b: bytes) -> str:
  return base64.b64encode(b).decode('ascii')

def _base64_decode(s: str) -> bytes:
    raw = base64.b64decode(s, validate=True)   # rejects bad chars / bad padding
    if base64.b64encode(raw).decode("ascii") != s:
        raise ValueError("non-canonical Base64")
    return raw


class BPETokenizer(TokenizerABC):
  PRETOKEN_RE = re.compile(rb"[A-Za-z]+|[^A-Za-z]")

  def __init__(self, threshold):
    self.threshold = threshold
    self.config = {
      "tokenizer": "bpe",
      "threshold": threshold
    }
    self.vocab_to_ind = {}
    self.ind_to_vocab = {}  # lazy evaluation
    self.merges = []

  def _pretokenize(self, text_bytes: bytes) -> list[bytes]:
    """
    Simple pre-tokenization logic to group consecutive letters as a word, any other 
    special characters or numbers are one word by each char.

    The reason is: Text matters more in plays. The separators are more important by 
    themselves. The numbers are not that important in generating plays. 
    """
    pieces = BPETokenizer.PRETOKEN_RE.findall(text_bytes)
    return pieces

  def _to_bytes(self, text: str):
    return text.encode(encoding="utf-8", errors='ignore')

  def _to_str(self, bytes_: bytes | bytearray):
    return bytes_.decode(encoding="utf-8", errors='ignore')

  def train(self, text: str):
    text_bytes = self._to_bytes(text)
    vocab = {
      b.to_bytes(): None 
      for b in sorted(text_bytes)
    } # Initial vocab is only raw utf-8 bytes.
      # Leverage dict's ordered feature to generate
      # deterministic vocab_to_ind id, given 
      # text_bytes.
    bytes_counter = Counter(self._pretokenize(text_bytes))  # count word bytes frequencies

    bytes_lists: list[tuple[list[bytes], int]] = [
      ([i.to_bytes() for i in word_bytes], freq) for
      word_bytes, freq in bytes_counter.items()
    ]
    merge_counts = self.threshold - len(vocab)
    if merge_counts < 0:
      logger.warning(f"No merges are allowed given threshold: \
                     {self.threshold} and raw vocab size: {len(vocab)}. \
                      BPE tokenizer degenerate to char level tokenizer.")

    self.merges = []

    for _ in tqdm(range(merge_counts), desc=f"Training BPE {merge_counts} merges", unit="merge"):
      pair_counter = Counter()

      for (sequence, freq) in bytes_lists:
        for i in range(len(sequence) - 1):
          pair_counter[(sequence[i], sequence[i+1])] += freq

      if len(pair_counter) == 0:
        # no pairs in all words are available to merge;
        # vocab size of self.threshold is unattainable
        logger.warning("No more merges available. \
                       Vocab size of threshold {self.threshold} is unattainable")
        break

      # Break ties in a deterministic fashion
      # Counter.most_common may be fine, but it tie breaking 
      # behavior depends on the insertion order, which may not
      # an ideal as a tie-breaking rule. 
      most_frequent_pair, _ = min(
        [(pair, freq) for pair, freq in pair_counter.items()],
        key=lambda item: (-item[1], item[0])
      )

      left = most_frequent_pair[0]
      right = most_frequent_pair[1]
      vocab[left + right] = None
      self.merges.append(most_frequent_pair)

      # merge the most frequent bytes pairs
      for j, (sequence, freq) in enumerate(bytes_lists):
        i = 0
        new_sequence = []
        while i < len(sequence):
          if (
            i + 1 < len(sequence) and
            left == sequence[i] and
            right == sequence[i+1]
          ):
            new_sequence.append(left + right)
            i += 2 
          else:
            new_sequence.append(sequence[i])
            i += 1
        bytes_lists[j] = (new_sequence, freq)

    # construct vocab_to_ind mapping
    self.vocab_to_ind = {}
    for i, t in enumerate(vocab.keys()):
      self.vocab_to_ind[t] = i

    self.ind_to_vocab = {v: k for k, v in self.vocab_to_ind.items()}

  def encode(self, text: str):
    text_bytes = self._to_bytes(text)
    bytes_lists = self._pretokenize(text_bytes)
    result = []

    for bytes_list in bytes_lists:
      if len(bytes_list) == 1:
        # only one bytes in the word, bytes_list is itself a bytes of only one byte
        result.append(self.vocab_to_ind[bytes_list])
        continue

      units = [b.to_bytes() for b in bytes_list]
      for left, right in self.merges:
        if len(units) < 2:
          break

        merged_units = []
        i = 0
        while i < len(units):
          if (
            i + 1 < len(units) and
            units[i] == left and
            units[i + 1] == right
          ):
            merged_units.append(left + right)
            i += 2
          else:
            merged_units.append(units[i])
            i += 1
        units = merged_units
      # merges
      token_ids = [self.vocab_to_ind[b] for b in units]
      result.extend(token_ids)

    return result

  def decode(self, token_ids):
    if not self.ind_to_vocab:
      self.ind_to_vocab = {v: k for k, v in self.vocab_to_ind.items()}
    bytes_ = b"".join([self.ind_to_vocab[i] for i in token_ids])
    return self._to_str(bytes_)

  @property
  def vocab_size(self):
    return len(self.vocab_to_ind)

  def save(self, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # save config
    config_path = path / "config.json"
    with open(config_path, "w", encoding="utf-8") as file:
      json.dump(self.config, file, indent=4)

    # save vocab_to_ind
    vocab_to_ind_path = path / "vocab_to_ind.json"
    encoded_vocab_to_ind = {
      # need to convert raw bytes to encodings before storing in json
      _base64_encode(k): v for k, v in self.vocab_to_ind.items()
    }
    with open(vocab_to_ind_path, "w", encoding="utf-8") as file:
      json.dump(encoded_vocab_to_ind, file, indent=4)

    # save merge list
    encoded_merges = [ [_base64_encode(l), _base64_encode(r)] for l, r in self.merges ]
    merges_path = path / "merges.json"
    with open(merges_path, "w", encoding="utf-8") as file:
      json.dump(encoded_merges, file, indent=4)

  @classmethod
  def load(cls, path):
    path = Path(path)
    # load config
    config_path = path / "config.json"
    with open(config_path, "r", encoding="utf-8") as file:
      config = json.load(file)

    # load vocab_to_ind
    vocab_to_ind_path = path / "vocab_to_ind.json"
    with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
      encoded_vocab_to_ind = json.load(file)
    vocab_to_ind = {
      _base64_decode(k): v for k, v in encoded_vocab_to_ind.items()
    }

    # load merge list
    merges_path = path / "merges.json"
    with open(merges_path, "r", encoding="utf-8") as file:
      encoded_merges = json.load(file)
    merges = [
      (_base64_decode(l), _base64_decode(r)) for l, r in encoded_merges
    ]

    t = cls(config["threshold"])
    t.vocab_to_ind = vocab_to_ind
    t.merges = merges
    return t


class BPETokenizerPlus(BPETokenizer):
  """BPE Tokenizer with slightly more nuanced pre-tokenization rule."""
  PRETOKEN_RE = re.compile(
      r"""'(?:s|t|re|ve|m|ll|d)\b"""
      r"""| ?[A-Za-z]+"""
      r"""| ?[0-9]+"""
      r"""| ?[^\sA-Za-z0-9]+"""
      r"""|\s+""",
      re.IGNORECASE,
  )

  def __init__(self, threshold, enable_timer=True):
    super().__init__(threshold, enable_timer)