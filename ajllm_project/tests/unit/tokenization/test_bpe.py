from pathlib import Path

from ajllm.tokenization.bpe_trainer import train_bpe
from ajllm.tokenization.serialization import load_vocab_and_merges, save_vocab_and_merges
from ajllm.tokenization.tokenizer import Tokenizer


def test_bpe_round_trip_with_unicode_and_special_token(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello hello 世界<|endoftext|>hello world\n", encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=270, special_tokens=["<|endoftext|>"])
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    save_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    loaded_vocab, loaded_merges = load_vocab_and_merges(vocab_path, merges_path)
    tokenizer = Tokenizer(loaded_vocab, loaded_merges, ["<|endoftext|>"])
    text = "hello 世界<|endoftext|>world"
    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    assert len(vocab) == 270
    assert tokenizer.encode("<|endoftext|>") == [269]


def test_streaming_encode_writes_uint32(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc abc abc\n", encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=260)
    tokenizer = Tokenizer(vocab, merges)
    output = tmp_path / "tokens.uint32"

    count = tokenizer.encode_file_to_disk(corpus, output)

    assert output.stat().st_size == count * 4
    assert count > 0


def test_parallel_bpe_training_matches_serial_training(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("alpha beta gamma<|endoftext|>\n" * 10) + ("delta epsilon zeta<|endoftext|>\n" * 10),
        encoding="utf-8",
    )

    serial_vocab, serial_merges = train_bpe(
        corpus,
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
        parallel=False,
    )
    parallel_vocab, parallel_merges = train_bpe(
        corpus,
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
        parallel=True,
        num_processes=2,
        chunk_size_bytes=64,
    )

    assert serial_vocab == parallel_vocab
    assert serial_merges == parallel_merges


def test_parallel_file_encoding_matches_serial_file_encoding(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("alpha beta gamma\n" * 20) + ("delta epsilon zeta\n" * 20), encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=280, parallel=False)
    tokenizer = Tokenizer(vocab, merges)
    serial_output = tmp_path / "serial.uint32"
    parallel_output = tmp_path / "parallel.uint32"

    serial_count = tokenizer.encode_file_to_disk(corpus, serial_output, chunk_size=64, parallel=False)
    parallel_count = tokenizer.encode_file_to_disk(
        corpus,
        parallel_output,
        chunk_size=64,
        parallel=True,
        num_processes=2,
    )

    assert serial_count == parallel_count
    assert serial_output.read_bytes() == parallel_output.read_bytes()
