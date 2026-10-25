import math
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, SequentialSampler

from datasets.dataset import LanguageModelingDataset
from utils.eval_utils import perplexity
from hyperparams import *


def test_perplexity():
  tokens = torch.tensor([9, 0, 1, 4, 5, 6, 3, 2, 8, 7, 10, 11])
  block_size = 3
  batch_size = 4
  vocab_size = 12

  idx = torch.randn(vocab_size)
  res = torch.zeros(batch_size, block_size, vocab_size)
  res = res + idx # assign the indices to the end

  class DummyModel(nn.Module):
    def __init__(self, vocab_size, block_size) -> None:
      super().__init__()

      self.vocab_size = vocab_size
      self.block_size = block_size
    
    def forward(self, x, y=None):
      [B, T] = x.shape
      return res[0:B, 0:T, :]

  dataset = LanguageModelingDataset(tokens, block_size)
  sampler = SequentialSampler(dataset)
  loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

  model = DummyModel(vocab_size, block_size)

  with torch.no_grad():
    total_loss = 0
    total_tokens = 0
    for batch in loader:
      inputs, labels = batch[0].contiguous(), batch[1].contiguous()
      inputs, labels = inputs, labels

      logits = model(inputs, None)  # Forward (Batch, Blocksize, Vocab size)
      B, T, V = logits.shape

      logits = logits.view(B*T, V)
      labels = labels.view(B*T)

      loss = nn.functional.cross_entropy(logits, labels, reduction='sum')

      total_loss += loss.item()
      total_tokens += labels.numel()

  expected_pp = math.exp(total_loss/total_tokens)
  pp = perplexity(model, loader)
  assert math.isclose(expected_pp, pp, rel_tol=1e-6), f"expected_pp: {expected_pp}; pp: {pp}"