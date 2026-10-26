from abc import ABC, abstractmethod

class PreprocessorABC(ABC):
  @classmethod
  @abstractmethod
  def preprocess(cls, text_input_path, text_output_path):
    return

  @classmethod
  def _store_corpus(cls, text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(text)
