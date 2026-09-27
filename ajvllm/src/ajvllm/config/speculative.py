"""Standard draft-model speculative decoding settings."""

from dataclasses import dataclass

from ajvllm.config.engine import require_int


@dataclass(frozen=True)
class SpeculativeConfig:
    enabled: bool = False
    draft_model: str = "model/Qwen2.5-0.5B-Instruct"
    num_draft_tokens: int = 3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("speculative.enabled must be a boolean")
        require_int("num_draft_tokens", self.num_draft_tokens, 1)
        if not isinstance(self.draft_model, str) or not self.draft_model:
            raise ValueError("draft_model must be a local checkpoint path")
