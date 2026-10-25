from preprocessors.preprocessors import ShakespearePlayDatasetPreprocessor


def get_preprocessor(dataset_name):
  if dataset_name.lower() == "shakespeare":
    return ShakespearePlayDatasetPreprocessor

  raise ValueError(f"Unknown dataset name: {dataset_name}")