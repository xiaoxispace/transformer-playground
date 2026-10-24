# Transformer Playground

A playground for people interested in transformers to experiment. 

# Example output


```
(shakespeare) ubuntu@193-122-144-209:~/dev/shakespeare-transformer$ python eval.py -m "data/model-final.pth"
Loading vocab ...
Vocab size:  83
Done loading vocab ...

Construct and load model ...
Construct and load model done.

================================
|| Begin generating contents: ||
================================

 puppets me falls, and down so in his cries
For thy fleet all edge me commended
With other to school of him thou dost quick crries.
If very doubt, Nature my father's feather.

SIR HUGH  Now, is is this from thus?

POLONIUS
I thank you faith-wilted, be dismission, get the other.
If thou best thou whoreson, with all appears
Blots, go with speak stock, or God's name
Doth she lie it it dower threater king?

GLOUCESTER
We'll said she, their gaze was fair Tito's vow.

TIMON
After did me than thou mean and children on so far
We have may blood) too.

[Enter Talbot.]


MESSENGER
Have you to York. Nurse! If he be marriages thing.
He loves the brother's name? Why, sir?
What means depart I'll like thee preserve.

POET  Where, bring him. By guess that you bear,
Not would say bondman.

BEROWNE  Not a pax the for who, of mother?

BOATSWAIN  Nay, if my lord of greater sake,
What is order,
And yet forgive shine own part. Therefore.

ULYSSES
That feather they offerent Caesar. I shall
Believe me soon pla
```

<details>
  <summary>Training Hyperparameters</summary>
  The hyperparameters used for the model generating the above outputs are:

  ```
  epoch = 100
  block_size = 64
  batch_size = 256
  dmodel = 256 
  learning_rate = 0.00001
  cuda_available = torch.cuda.is_available()
  num_of_encoder_layers = 3
  num_of_heads = 4
  decoder_only = True
  ```

</details>

Note: 
- The model generating the example output is not saved as one of the checkpoints. A similar checkpoint can be found under `checkpoints/ctx64`, where the learning rate is by default, `0.0003`.
- The model output was truncated by length, as to keep the simplicity, tokenizer does not include any of the special tokens such as `BOS` (the beginning of a sequence), `EOS` (end of sequence), `UNK` unknown token, etc.

# Setup

## Install Python Environment

1. Install pyenv from [the pyenv public repo](https://github.com/pyenv/pyenv#a-getting-pyenv). Specifically, for linux, 
    1. Update and install system dependencies
        ```bash
        sudo apt update
        sudo apt install -y make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev curl git \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev
        ```
    2. Run `curl -fsSL https://pyenv.run | bash` to install pyenv
    3. Setup pyenv for shell with command
        ```bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        ```

        Note: If this is a Mac, replace `.bashrc` with `.zshrc` in all three commands above.
    4. Restart shell with command `exec "$SHELL"`
2. Install Python and the required dependencies
    1. Run `pyenv install 3.12.15`
    2. Create the virtual environment `pyenv virtualenv 3.12.15 transformer-playground`
    3. Install the required dependencies `python -m pip install -r requirements.txt`
    4. Note: In an environment powered by CPU such as Mac without MPS or Linux without CUDA, please use the requirements file `requirements-cpu.txt`, which installs native `torch` and `torchvision` pacakges instead of their CUDA counterparts.
3. Execute training, eval or generation scripts. 
    - To train model, run `python train.py -e <epoch number> --tokenizer-path <tokenizer saved path> --dataset-path <path to dataset text file> --enable-mixed-precision`. `--enable-wandb` is optional provided environment variable `WANDB_API_KEY` is available to upload training statistics to WandB.
      Note: To use tmux with, for example, training script: 
        ```bash
        tmux new-session -d -s train 'python train.py -e 100 --tokenizer-path "data/bpe_tokenizer" --dataset-path "data/output.txt" --enable-wandb --enable-mixed-precision'
        ```
    - To evaluate model on the test dataset, run `python eval.py -m <path to the model>`
    - To genearate response, run `python gen.py -m <path to the model>`

## Default Parameters

The default parameters is as follows. Note that `bpe_vocab_threshold` is only valid when `tokenizer` is `bpe`. 
<details>
  <summary>Hyperparameters Specification</summary>

  ```python
  config = {
    # model parameters
    "block_size": 256,
    "dmodel": 256,
    "dropout": 0.1,
    "num_of_decoder_layers": 3,
    "num_of_encoder_layers": 3,
    "num_of_heads": 4,
    "decoder_only": True,
    # training parameters
    "batch_size": 256,
    "cuda_available": torch.cuda.is_available(),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "learning_rate": 3e-4,
    # tokenizer parameters
    "tokenizer": "char",
    "bpe_vocab_threshold": 500,
    # dataset parameters
    "dataset": "shakespeare"
  }
  ```
</details>

## Tokenizer Storage

By default, each tokenizer is stored with two files `config.json` and `vocab_to_ind.json` under `data/tokenizers` directory such that:
- `config.json`: records the basic configuration of the tokenizer.
- `vocab_to_ind.json`: maps from each token to its token index. 

If additional files are required to define the tokenizer, they will be stored together with the above two files.

## Checkpoints Storage

Checkpoints are stored under `checkpoints`.

If model parameters differ in the checkpoint, it will be noted by the checkpoint directory name. For example, `ctx64-dmodel128-lr1e-3` suggests the following updates to the above default parameters:
- `block_size`: set to 64
- `dmodel`: set to 128
- `learning_rate`: set to 1e-3

Similarly, `ctx64-dcdr1` suggests the model context (`block_size`) is 64, the number of decoder layers (`dcdr`) is 1. `bpe-ctx64-500threshold` suggests BPE tokenizer is used with 500 threshold and the model context is 64.

# Unittests

- To run unittests of the folder after the environment is installed, execute `pytest .`
- To run performance/profiling tests that are marked "skipped", execute `pytest -m skip`

# Reference
The complete text of Shakespeare's work can be found in [Folger Shakespeare Library](https://www.folger.edu/explore/shakespeares-works/all-works/). 
