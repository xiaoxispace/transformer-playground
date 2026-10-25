import torch

from torch.utils.data import DataLoader

class LanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y


class ShakespearePlayDataset(LanguageModelingDataset):
  def __init__(self, tokens, block_size):
    super().__init__(tokens, block_size)


def prepare_dataset_loader(tokens, block_size, batch_size, train_end, val_end):
  n = len(tokens)

  train_end = int(train_end * n)
  val_end = int(val_end * n)

  train_tokens = tokens[:train_end]
  val_tokens = tokens[train_end:val_end]
  test_tokens = tokens[val_end:]

  train_dataset = ShakespearePlayDataset(train_tokens, block_size)
  val_dataset = ShakespearePlayDataset(val_tokens, block_size)
  test_dataset = ShakespearePlayDataset(test_tokens, block_size)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader, test_loader