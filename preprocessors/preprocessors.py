import logging

from .preprocessor_abc import PreprocessorABC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess")


class ShakespearePlayDatasetPreprocessor(PreprocessorABC):
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
  def _store_corpus(cls, text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(text)

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
