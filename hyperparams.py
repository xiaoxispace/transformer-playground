"""This file contains the hyperparameters related to model architecture, tokenizer, and training."""
import torch


config = {
  # model parameters
  "block_size": 256,
  "batch_size": 256,
  "dmodel": 256,
  "dropout": 0.1,
  "num_of_decoder_layers": 3,
  "num_of_encoder_layers": 3,
  "num_of_heads": 4,
  "decoder_only": True,
  # training parameters
  "cuda_available": torch.cuda.is_available(),
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "learning_rate": 3e-4,
  # tokenizer parameters
  "tokenizer": "bpe",
  "bpe_vocab_threshold": 500,
  # dataset parameters
  "dataset": "shakespeare",
  "train_end": 0.7,
  "val_end": 0.9,
}