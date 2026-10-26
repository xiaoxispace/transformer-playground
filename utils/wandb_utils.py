import wandb


def normalize_wandb_config(config):
  wandb_config = {}
  for k, v in config.items():
    wandb_config[k] = str(v)

  if wandb_config["tokenizer"] == "char":
    del wandb_config["bpe_vocab_threshold"]

  return wandb_config


def setup_wandb(config, enable_wandb, dataset_names):
  wandb_config = normalize_wandb_config(config)
  wandb_config["datasets"] = dataset_names
  if enable_wandb:
    wandb.init(
      project="shakespeare-transformer",
      config=wandb_config
    )
    run_name = wandb.run.name
  else:
    print("Disable wandb")
    wandb.init(mode="disabled")
    run_name = None

  return run_name
