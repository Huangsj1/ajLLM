"""SFT names for the labelled causal-LM training loop."""

from ajllm.training.pretrainer import PretrainConfig, Pretrainer

# The optimizer, AMP, checkpoint, distributed synchronization, and evaluation
# mechanics are identical.  SFT differs at the dataset boundary: labels mask
# prompts and retain only assistant completions.
SFTConfig = PretrainConfig
SFTTrainer = Pretrainer

__all__ = ["SFTConfig", "SFTTrainer"]
