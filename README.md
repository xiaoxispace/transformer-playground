# Transformer Playground

A playground for people interested in transformers to experiment. 

# Quick Start

## Generate Plays

To generate plays using the project's checkpoints, execute the command

```bash
python gen.py -m <path to a checkpoint .pth file>
```

The number of tokens to generate and the prompt can be specified by the parameter `-l <an integer>` and `-s <a string>`. 

<details>

To generate using the checkpoint `checkpoints/ctx384-3e-5/best_model_ctx384_d256_L3_H4_lr3e-05_dropout1e-01_e47.pth` with length 2000 tokens and prompt "Hello", one can use the command `python gen.py -m "checkpoints/ctx384-3e-5/best_model_ctx384_d256_L3_H4_lr3e-05_dropout1e-01_e47.pth" -l 2000 -s "Hello"`.

An example output is shown below. 

<summary>Generation example with model of 384 context size</summary>

```
Model config:  {
  "block_size": "384",
  "dmodel": "256",
  "dropout": "0.1",
  "num_of_decoder_layers": "3",
  "num_of_encoder_layers": "3",
  "num_of_heads": "4",
  "decoder_only": "True",
  "vocab_size": "83"
}
|| Begin generating contents: ||
w.

LEAR
Call this things in my dimate of alungbrace
And we have dukedom of Guard thee fingers.
Which doth stiny a brough in sway, set here buard
Where he haste ward us find her faithful men?
But be rest, and not did groans, for eyes,
Did be sword, one foreign receive that morning
That have all knows, the true distance,
Franciscience. Fetch double spurse
As daughter business. I am of pay live with
let on my left fools examine, and those and,
And that which he searest the right
That can such a triumphant our dukes were as guard,
Which he shall have sway oswed by his looks?

NORTHUMBERLAND
But think of them wears.
[To Servant exits.]

Scene 5
[Enter Hector is of York Hero, are Peter Queen,
Clarence, Clifford, looked a back of spirit,
And will unworthy Marina, if peace!

[Enter Antonyes.]


ARCHBISHOP  Sha, she takes a prize of war,
Which I mights many be done thanked of your
fellow him. Even I might enemies. [They all the Near.] What have shall?

ROMEO
With these part; what cruel thee th
```

</details>

For details to setup an environment, refer to the [Install Python Environment](# install-python-environment) section.

Note: tokenizer does not include any of the special tokens such as `BOS` (the beginning of a sequence), `EOS` (end of sequence), `UNK` unknown token, etc. So the start and end of the generation is controlled by input prompt and fixed generation length respectively. 

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

# Training Setup

The training pipeline accepts a tokenizer and a seuqence of corpus paths as input. The tokenizer will tokenize the text corpus, split and combine them into training, validation, and test datasets. 

## Download and Preprocess Datasets

Currently, there are two datasets available for training Transformers, Shakespeare's play texts (dataset name: `shakespeare`) and Shakespeare's contemporary plays (dataset name: `emed`). 

Each dataset is stored under `data/datasets/<dataset name>`. The Shakespeare dataset has its raw data in the repo. For EMED dataset, a download script and a list of links to the plays are provided for not storing large set of XML files in the repo. Please execute `./download_shc.sh links.txt plays/` under `data/datasets/emed` to have the XML files ready for preprocessing. 

To preprocess a dataset, execute `python preprocess.py --dataset <dataset name>` at the project root. The concatenated text corpus will be produced as `output.txt` under each dataset's directory. The text file `output.txt` can be used to train tokenizers or a Transformer model after being tokenized. 

## Default Parameters

The default parameters for training tokenizers or Transformer models exist in `hyperparams.py`. Note that `bpe_vocab_threshold` is only valid when `tokenizer` is `bpe`. 

## Training Tokenizers

Given `output.txt` or a sequence of text files, we can train a tokenizer by executing `python train_tokenizer.py --dataset-paths {sequence of paths to text files}`.

### Tokenizer Storage

By default, each tokenizer is stored with two files `config.json` and `vocab_to_ind.json` under `data/tokenizers` directory such that:
- `config.json`: records the basic configuration of the tokenizer.
- `vocab_to_ind.json`: maps from each token to its token index. 

If additional files are required to define the tokenizer, they will be stored in conjunction with the above two files.

## Training and evaluating Transformer Models

- To train model, run `python train.py -e <epoch number> --tokenizer-path <tokenizer saved path> --dataset-paths <a sequence of paths to dataset text file> --enable-mixed-precision`. `--enable-wandb` is optional provided environment variable `WANDB_API_KEY` is available to upload training statistics to WandB.
  Note: To use tmux with, for example, training script: 
    ```bash
    tmux new-session -d -s train 'python train.py -e 100 --tokenizer-path "data/bpe_tokenizer" --dataset-path "data/output.txt" --enable-wandb --enable-mixed-precision'
    ```
- To evaluate model on the test dataset, run `python eval.py -m <path to the model>`

### Checkpoints Storage

Checkpoints are stored under `checkpoints`.

If model parameters differ in the checkpoint, it will be noted by the checkpoint directory name. For example, `ctx64-dmodel128-lr1e-3` suggests the following updates to the above default parameters:
- `block_size`: set to 64
- `dmodel`: set to 128
- `learning_rate`: set to 1e-3

Similarly, `ctx64-dcdr1` suggests the model context (`block_size`) is 64, the number of decoder layers (`dcdr`) is 1. `bpe-ctx64-500threshold` suggests BPE tokenizer is used with 500 threshold and the model context is 64.

## Unittests

- To run unittests of the folder after the environment is installed, execute `pytest .`
- To run performance/profiling tests that are marked "skipped", execute `pytest -m skip`

# Reference

This project uses early modern English dramatic texts from the following sources:

- **Folger Shakespeare Library:** Shakespeare's plays were obtained from [Folger Shakespeare Library's digital text collection](https://www.folger.edu/explore/shakespeares-works/download/). I gratefully acknowledge the Folger Shakespeare Library and the editors of Folger Digital Texts for providing authoritative, machine-readable editions of Shakespeare's works.
- **Early Modern English Drama (EMED)**: A digital anthology administered by the Folger Institute and Folger Shakespeare Library. I acknowledge EMED for providing access to the SHC corpus files through its corpus download platform.
- **Shakespeare His Contemporaries (SHC):** A digital text corpus led by Martin Mueller at Northwestern University. I gratefully acknowledge the SHC project and its contributors for making the corpus available for research and educational use.

The source texts were processed for Transformer-based language-model training. For details, refer to the comments of the preprocessor classes: 
- For processing on the Shakespeare's texts, the processing is documented in `ShakespearePlayDatasetPreprocessor`
- For SHC texts, the processing is documented in `EMEDDatasetPreprocessor` and `_SHCPreprocessor`

The SHC and Folger Shakespeare texts are subject to their respective licensing terms. Thus, this project should not be used for commercial purposes. Please consult the original sources for details.

Source links

- Folger Shakespeare texts: https://www.folger.edu/explore/shakespeares-works/download/
- EMED data and download documentation: https://emed.folger.edu/research-resources/data