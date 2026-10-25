from abc import ABC, abstractmethod

class PreprocessorABC(ABC):
  @classmethod
  @abstractmethod
  def preprocess(cls, text_input_path, text_output_path):
    return
