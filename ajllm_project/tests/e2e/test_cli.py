from ajllm.cli import build_parser


def test_cli_parses_every_domain() -> None:
    parser = build_parser()

    assert parser.parse_args(["tokenizer", "train", "-c", "config.yaml"]).command == "train"
    assert parser.parse_args(["lm", "sweep", "-c", "config.yaml"]).command == "sweep"
    assert parser.parse_args(["model", "report", "-c", "config.yaml"]).command == "report"
