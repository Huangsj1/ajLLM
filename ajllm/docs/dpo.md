# Direct Preference Optimization (DPO)

## 1. Objective

DPO starts from an SFT checkpoint and keeps an identical frozen reference
policy \(\pi_{\mathrm{ref}}\). For prompt \(x\), preferred completion
\(y_w\), and rejected completion \(y_l\), the implemented objective is:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}}) =
-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)\right] + \mathcal{L}_{\mathrm{MoE\ aux}}.
$$

The trainer sums completion-token log-probabilities per branch, calculates the
chosen/rejected policy-reference margins, and applies `-logsigmoid`. The
reference is frozen for the entire run.

## 2. Dataset

`data/dpo.jsonl` contains one pair of conversations per line. The shared prefix
is the prompt; both branches must end in an assistant response.

```json
{
  "chosen": [
    {"role": "user", "content": "continue"},
    {"role": "assistant", "content": "As Recharge Retreats grows, we plan to expand our team..."}
  ],
  "rejected": [
    {"role": "user", "content": "continue"},
    {"role": "assistant", "content": "A. Scaling: 1. Offer a franchise model..."}
  ]
}
```

Only the final assistant completion in each branch contributes to its sequence
log-probability. MiniMind message formatting is shared with SFT and generation.

## 3. Prerequisites

```bash
uv sync --extra dev
```

DPO needs a completed matching SFT checkpoint for both the initial policy and
immutable reference. The supplied paths are:

```text
output/sft/dense/step_00100000.pt
output/sft/moe/step_00159670.pt
```

No reward-model download is required.

## 4. Train

```bash
# Dense: output/dpo/dense/step_00003219.pt
uv run python -m ajllm.workflows.dpo --config configs/dpo/dense.yaml

# Top-1 MoE: output/dpo/moe/step_00003219.pt
uv run python -m ajllm.workflows.dpo --config configs/dpo/moe.yaml
```

Keep `sft_checkpoint` fixed when resuming. `resume_from` restores only the
trainable policy, optimizer, sampler position, and completed step.

## 5. Plot

```bash
uv run python -m ajllm.workflows.plot_training --run output/dpo/dense --smooth 20
uv run python -m ajllm.workflows.plot_training --run output/dpo/moe --smooth 20
```

The common workflow plots `total_loss`. `metrics.jsonl` additionally records
`chosen_reward`, `rejected_reward`, `reward_margin`, and
`preference_accuracy` for preference diagnostics.

## 6. Generate

Use a DPO checkpoint as an ordinary portable policy checkpoint:

```bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/dpo/dense/step_00003219.pt \
  --messages-json assets/prompt/chat_messages.json --open-thinking

uv run python -m ajllm.workflows.generate \
  --checkpoint output/dpo/moe/step_00003219.pt \
  --messages-json assets/prompt/chat_messages.json --open-thinking

uv run python -m ajllm.workflows.generate \
  --checkpoint output/dpo/dense/step_00003219.pt \
  --prompt-file assets/prompt/continuation.txt
```

## 7. Multi-GPU

DPO uses the common causal-LM workflow, so its policy and frozen reference can
be wrapped with the same FSDP, TP, EP, or TP×EP strategies as the matching
architecture. No dedicated DPO parallel YAML is shipped: copy a dense or MoE
DPO YAML, preserve its SFT checkpoint, and add the desired parallel block from
[parallel training](parallel_training.md). Then launch one process per GPU:

```bash
# Example: copied dense DPO YAML with parallel.tp_size: 4, parallel.ep_size: 1
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.dpo --config configs/dpo/dense_tp4.yaml
```

FSDP cannot be combined with TP/EP. Resume only with the same topology and keep
all rank-local optimizer files beside the main checkpoint.
