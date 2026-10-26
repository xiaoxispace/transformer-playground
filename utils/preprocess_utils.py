from preprocessors.preprocessors import ShakespearePlayDatasetPreprocessor, EMEDDatasetPreprocessor


def get_preprocessor(dataset_name):
  if dataset_name == "shakespeare":
    return ShakespearePlayDatasetPreprocessor
  elif dataset_name == "emed":
    return EMEDDatasetPreprocessor

  raise ValueError(f"Unknown dataset name: {dataset_name}")