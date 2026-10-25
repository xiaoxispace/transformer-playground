import pytest

from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer
from utils.tokenizer_utils import create_tokenizer, load_tokenizer

def test_create_tokenizer():
  char_config = {"tokenizer": "char"}
  bpe_config = {"tokenizer": "bpe", "threshold": 10}

  t = create_tokenizer(char_config)
  assert isinstance(t, CharTokenizer)

  t = create_tokenizer(bpe_config)
  assert isinstance(t, BPETokenizer)


def test_load_tokenizer(cache):
  # need pytest's builtin fixture cache object
  pytest_cache_path = cache._cachedir
  corpus = "This is a very long and profoundly philosophical sentence."
  char_config = {"tokenizer": "char"}
  bpe_config = {"tokenizer": "bpe", "threshold": 10}
  c_t = create_tokenizer(char_config)
  bpe_t = create_tokenizer(bpe_config)
  c_t.train(corpus)
  bpe_t.train(corpus)

  c_t_dir = "c_t-load-test"
  c_t_path = pytest_cache_path / c_t_dir
  c_t.save(c_t_path)
  bpe_t_dir = "bpe_t-load-test"
  bpe_t_path = pytest_cache_path / bpe_t_dir
  bpe_t.save(bpe_t_path)

  loaded_c_t = load_tokenizer(c_t_path)
  loaded_bpe_t = load_tokenizer(bpe_t_path)

  assert loaded_c_t.config == c_t.config
  assert loaded_c_t.vocab_to_ind == c_t.vocab_to_ind
  assert loaded_bpe_t.config == bpe_t.config
  assert loaded_bpe_t.vocab_to_ind == bpe_t.vocab_to_ind
  assert loaded_bpe_t.merges == bpe_t.merges


def test_char_tokenizer():
  corpus = "This is a very long and profoundly philosophical sentence."
  t = CharTokenizer()
  t.train(corpus)

  token_ids = t.encode(corpus)
  decoded_corpus = t.decode(token_ids)
  assert len(corpus) == len(decoded_corpus), "text corpus should have same length"
  for i in range(len(corpus)):
    assert corpus[i] == decoded_corpus[i], "decoded corpus should have the same content"


def test_char_save_and_load(cache):
  # need pytest's builtin fixture cache object
  pytest_cache_path = cache._cachedir
  corpus = "Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  t = CharTokenizer()
  t.train(corpus)
  char_tokenizer_save_test_path = "char-save-test"
  temp_path = pytest_cache_path / char_tokenizer_save_test_path
  t.save(temp_path)
  loaded_t = CharTokenizer.load(temp_path)
  assert t.vocab_size == loaded_t.vocab_size
  assert t.vocab_to_ind == loaded_t.vocab_to_ind
  assert t.config == loaded_t.config


@pytest.mark.skip
def test_bpe_train_large_corpus():
  """
  Use for performance profiling & validation.
  
  Marked with "slow" to avoid unnecessary runs.
  """
  threshold = 500
  t = BPETokenizer(threshold)
  with open("data/output.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

  import time
  start = time.perf_counter()
  t.train(corpus)
  elapsed = time.perf_counter() - start
  print(f"Training with threshold {threshold} takes {elapsed:.4f}s")  # 60~80 seconds

  start = time.perf_counter()
  res = t.encode(corpus)
  elapsed = time.perf_counter() - start
  print(f"Encode corpus of length {len(corpus)} takes {elapsed:.4f}s") # ~230 seconds

  start = time.perf_counter()
  decoded_corpus = t.decode(res)
  elapsed = time.perf_counter() - start
  print(f"Decode corpus of length {len(corpus)} takes {elapsed:.4f}s")
  assert corpus == decoded_corpus


def test_bpe_pretokenize():
  text_bytes = b"Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  expected = [
    b"Hello", b",", b" ", b"world", b"!", b" ",
    b"There", b"'", b"s", b" ", b"no", b" ", b"stop", b" ",
    b"_", b"in", b" ", b"word", b"!", b" ",
    b"^", b"&", b"*", b")", b"(", b")", b"%", b"$", b"#", b"@", b"!", b"~", b"`", b"'", b"\"",
    b";", b":", b"/", b"?", b".", b">", b",", b"<", b"\\", b"|", b"-", b"_", b"=", b"+", b"\t", b"\n"
  ]
  t = BPETokenizer(50)
  res = t._pretokenize(text_bytes)
  assert res == expected


def test_bpe_train():
  # corpus has 45 characters
  corpus = "Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  t = BPETokenizer(50)  # threshold is 50 so there will be 5 merges
  t.train(corpus)
  assert len(t.vocab_to_ind) == 50
  for c in corpus:
    # assert each byte of the original corpus is in the vocab
    assert c.encode() in t.vocab_to_ind  


def test_bpe_train_large_threshold():
  corpus = "Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  t = BPETokenizer(70)
  t.train(corpus)
  res = t.encode(corpus)
  test_corpus = t.decode(res)
  assert corpus == test_corpus


def test_bpe_encode_decode():
  corpus = "Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  t = BPETokenizer(70)
  t.train(corpus)
  res = t.encode(corpus)
  test_corpus = t.decode(res)
  assert corpus == test_corpus


def test_degenerate_bpe_encode_decode():
  corpus = "Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  t = BPETokenizer(3) # this is effectively a char tokenizer
  t.train(corpus)
  res = t.encode(corpus)
  test_corpus = t.decode(res)
  assert corpus == test_corpus


def test_bpe_save_and_load(cache):
  # need pytest's builtin fixture cache object
  pytest_cache_path = cache._cachedir
  # the corpus has 45 chars; use 60 as threshold to test saving "merged bytes" behavior
  corpus = "Hello, world! There's no stop _in word! ^&*)()%$#@!~`'\";:/?.>,<\\|-_=+\t\n"
  t = BPETokenizer(60)
  t.train(corpus)
  bpe_tokenizer_save_test_path = "bpe-save-test"
  temp_path = pytest_cache_path / bpe_tokenizer_save_test_path
  t.save(temp_path)
  loaded_t = BPETokenizer.load(temp_path)
  assert t.vocab_size == loaded_t.vocab_size
  assert t.vocab_to_ind == loaded_t.vocab_to_ind
  assert t.config == loaded_t.config