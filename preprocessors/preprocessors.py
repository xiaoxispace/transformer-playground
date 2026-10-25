import logging

from tqdm import tqdm
from pathlib import Path

from .emed_subpreprocessors import _SHCPreprocessor
from .preprocessor_abc import PreprocessorABC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess")


class ShakespearePlayDatasetPreprocessor(PreprocessorABC):
  """
  This is the preprocessor for Shakespeare datasets. The preprocessor completes
  the following tasks:

  1. Reads all the plays under data/shakespeares-works_TXT_FolgerShakespeare
  2. Strip all the comments before "Characters in the Play" (if any)
  3. Remove all the lines of "=" separator in each play
  4. Concatenate all the players into one single txt file

  Note: The headers of plays without "Characters in Play" are removed manually.
  """
  @classmethod
  def _read_play_file(cls, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      text = f.read()
    return text

  @classmethod
  def _remove_headers(cls, text):
    """Remove headers before the character introduction 'Characters in the Play'"""
    start_ind = text.find("Characters in the Play")
    if start_ind >= 0:
      return text[start_ind:]
    else:
      # headers already removed manually.
      return text

  @classmethod
  def _remove_separator_lines(cls, s):
    """Remove separator line of '='. """
    return '\n'.join(
      line for line in s.splitlines()
      if not line or set(line) != {'='}
    )

  @classmethod
  def preprocess(cls, text_input_path, text_output_path):
    data_path = text_input_path
    output_path = text_output_path
    output = ""
    for file_path in sorted(data_path.glob("*.txt")):
      logger.info(f"Begin reading {file_path}")
      text = cls._read_play_file(file_path)
      text = cls._remove_headers(text)
      text = cls._remove_separator_lines(text)
      output = output + '\n' + text
      logger.info("Done appending.")
    output = output[1:]  # remove the first \n
    cls._store_corpus(output, output_path)


class EMEDDatasetPreprocessor(PreprocessorABC):
  """This processor delegates XML parsing task to _SHCPreprocessor."""

  @classmethod
  def preprocess(cls, text_input_path, text_output_path):
    data_path = Path(text_input_path)
    output_path = Path(text_output_path)
    output_parts = []
    for file_path in tqdm(sorted(data_path.glob("*.xml")), desc="Process EMED datasets", unit="files"):
      text = _SHCPreprocessor._preprocess_shc(file_path)
      if text.strip():
        output_parts.append(text)
    output = "\n\n".join(output_parts)
    cls._store_corpus(output, output_path)