import torch

from torch.utils.data import DataLoader

from datasets.dataset import LanguageModelingDataset


def load_and_split_dataset_tensor(dataset_path, tokenizer, train_end, val_end):
  with open(dataset_path, 'r', encoding='utf-8') as file:
    raw_tokens = file.read()
  token_ids = tokenizer.encode(raw_tokens)
  token_tensor = torch.tensor(token_ids).to(torch.long)
  n = len(token_tensor)
  train_end = int(train_end * n)
  val_end = int(val_end * n)

  training_set_tensors = token_tensor[:train_end]
  val_set_tensors = token_tensor[train_end:val_end]
  test_set_tensors = token_tensor[val_end:]
  return training_set_tensors, val_set_tensors, test_set_tensors


def load_and_combine_dataset_tensors(dataset_paths, tokenizer, train_end, val_end):
  train_t_list = []
  val_t_list = []
  test_t_list = []
  for p in dataset_paths:
    train_t, val_t, test_t = load_and_split_dataset_tensor(p, tokenizer, train_end, val_end)

    train_t_list.append(train_t)
    val_t_list.append(val_t)
    test_t_list.append(test_t)

  combined_train_tensors = torch.concat(train_t_list, dim=0)
  combined_val_tensors = torch.concat(val_t_list, dim=0)
  combined_test_tensors = torch.concat(test_t_list, dim=0)

  return combined_train_tensors, combined_val_tensors, combined_test_tensors


def prepare_dataset_loader(dataset_paths, tokenizer, train_end, val_end, block_size, batch_size):
  training_set_tensors, val_set_tensors, test_set_tensors = load_and_combine_dataset_tensors(
    dataset_paths, tokenizer, train_end, val_end
  )
  train_dataset = LanguageModelingDataset(training_set_tensors, block_size)
  val_dataset = LanguageModelingDataset(val_set_tensors, block_size)
  test_dataset = LanguageModelingDataset(test_set_tensors, block_size)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader, test_loader