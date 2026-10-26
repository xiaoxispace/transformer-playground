
import argparse
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from pathlib import Path
from tqdm import tqdm

from utils.dataset_utils import prepare_dataset_loader
from hyperparams import config
from models.transformer import Transformer
from utils.model_utils import make_model_saver, extract_model_config
from utils.tokenizer_utils import load_tokenizer
from utils.wandb_utils import setup_wandb


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    save_model_fn,
    epochs,
    device='cuda',
    use_bf16=False,
  ):
  # Define loss function and optimizer
  # Training loop
  train_loss_history = []
  test_loss_history = []
  val_loss_history = []
  epoch_sequence = []

  min_val_loss = torch.inf
  min_val_loss_epoch = 0
  model_of_min_val_loss = None
  best_model_path = None

  for epoch in range(epochs):
    # training
    log_registry = {}
    training_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
      model.train()
      for batch in train_loader:
        inputs, labels = batch[0].contiguous(), batch[1].contiguous()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients

        with torch.autocast(
          device_type=device,
          dtype=torch.bfloat16,
          enabled=use_bf16,
        ):
          logits = model(inputs, None)
          B, T, C = logits.shape
          logits = logits.view(B * T, C)
          labels = labels.view(B * T)

          loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        training_loss += loss.item() * B
        pbar.update(1)
    training_loss = training_loss / len(train_loader.dataset)
    train_loss_history.append(training_loss)
    log_registry['train_loss'] = training_loss

    # validation & testing
    with torch.no_grad():
      model.eval()
      val_loss = 0.0
      for batch in val_loader:
        inputs, labels = batch[0].contiguous(), batch[1].contiguous()
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs, None)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        labels = labels.view(B * T)

        loss = criterion(logits, labels)
        val_loss += loss.item() * B

      test_loss = 0.0
      for batch in test_loader:
        inputs, labels = batch[0].contiguous(), batch[1].contiguous()
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs, None)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        labels = labels.view(B * T)

        loss = criterion(logits, labels)
        test_loss += loss.item() * B

    val_loss = val_loss / len(val_loader.dataset)
    val_loss_history.append(val_loss)
    log_registry['val_loss'] = val_loss
    test_loss = test_loss / len(test_loader.dataset)
    test_loss_history.append(test_loss)
    log_registry['test_loss'] = test_loss
    wandb.log(log_registry)

    epoch_sequence.append(epoch + 1)

    print(f'Epoch {epoch + 1}/{epochs}: train loss {training_loss}, validation loss {val_loss}, test loss {test_loss_history[-1]}')

    if epoch % 5 == 0 or epoch == epochs - 1:
      # save every 5 epochs or reaches the end of epoch
      intermediate_model_path, _ = save_model_fn(model, epoch)
      print(f"  Intermediate model at epoch {epoch} saved to {intermediate_model_path}")

    if min_val_loss > val_loss:
      print(f"Epoch {epoch + 1} reaches new minimum validation loss.")
      model_of_min_val_loss = copy.deepcopy(model).cpu()
      min_val_loss_epoch = epoch
      min_val_loss = val_loss

      if best_model_path:
        best_model_path.unlink(missing_ok=True)
        print(f"  Removed previous best model {best_model_path.name}")

      best_model_path, _ = save_model_fn(model_of_min_val_loss, min_val_loss_epoch, is_best_model=True)
      print(f"  Latest best model saved to {best_model_path}")

  if model_of_min_val_loss and best_model_path:
    print(f"Model with lowest validation loss achieved at epoch {min_val_loss_epoch}.")
    print(f"Best model name: {best_model_path.name}")

  print('Training complete!')

  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                  prog='shakespeare-training',
                  description='pretrain shakespeare transformer')
  parser.add_argument('-e', '--epochs', default=100, type=int)
  parser.add_argument('--tokenizer-path', type=str)
  parser.add_argument('--dataset-paths', nargs='+', type=str)
  parser.add_argument('--enable-wandb', action='store_true')
  parser.add_argument('--enable-mixed-precision', action='store_true')
  parser.add_argument('--deterministic-training', action='store_true')

  args = parser.parse_args()
  epochs = args.epochs
  tokenizer_path = args.tokenizer_path
  dataset_paths = args.dataset_paths
  enable_wandb = args.enable_wandb
  enable_mixed_precision = args.enable_mixed_precision
  deterministic_training = args.deterministic_training

  train_end = config['train_end']
  val_end = config['val_end']

  block_size = config['block_size']
  batch_size = config['batch_size']

  learning_rate = config["learning_rate"]
  device = config["device"]
  use_bf16 = (
    enable_mixed_precision
    and device == "cuda"
    and torch.cuda.is_bf16_supported()
  )

  dataset_names = [Path(p).parent for p in dataset_paths]
  run_name = setup_wandb(config, enable_wandb, dataset_names)

  print("Hello World!")
  print(f"Loading tokenizer from path {tokenizer_path} ...")
  tokenizer = load_tokenizer(tokenizer_path)
  config["vocab_size"] = tokenizer.vocab_size
  print("Loading tokenizer done.")

  print("Preparing dataset loader ...")
  train_loader, val_loader, test_loader = prepare_dataset_loader(dataset_paths, tokenizer, train_end, val_end, block_size, batch_size)
  print("Dataset loader ready.")

  model = Transformer(**extract_model_config(config)).to(device)
  config_display = {k: str(v) for k, v in config.items()}
  print("Model trainable parameters: ", model.count_trainable_parameters())
  print(f"Mixed precision enabled: {use_bf16}")
  print("CUDA device count: ", torch.cuda.device_count())
  print("CUDA available: ", torch.cuda.is_available())
  print("Config: ", json.dumps(config_display, indent=2))
  print("Epochs: ", epochs)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  model = train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    save_model_fn=make_model_saver(config, tokenizer, run_name=run_name),
    epochs=epochs,
    device=device,
    use_bf16=use_bf16
  )
