# Group Relative Policy Optimization (GRPO)

## 1. Objective

GRPO is online RLAIF after SFT. For each prompt \(x_i\), the policy samples
\(G\) completions; the frozen Skywork reward model scores them and advantages
are normalized inside that group:

$$
A_{i,j}=\frac{R_{i,j}-\mu_i}{\sigma_i+\epsilon_A}.
$$

For completion token \(y_{i,j,t}\), fixed behavior policy
\(\pi_{\mathrm{old}}\), and frozen SFT reference \(\pi_{\mathrm{ref}}\),
the implemented token objective is:

$$
\ell_{i,j,t}(\theta) =
-\min\left(r_{i,j,t}(\theta)A_{i,j},
\operatorname{clip}(r_{i,j,t}(\theta),1-\varepsilon,1+\varepsilon)A_{i,j}\right)
+\beta D_{\mathrm{KL},i,j,t},
$$

$$
r_{i,j,t}=\frac{\pi_\theta(y_{i,j,t}\mid x_i,y_{i,j,<t})}
{\pi_{\mathrm{old}}(y_{i,j,t}\mid x_i,y_{i,j,<t})},\qquad
D_{\mathrm{KL}}=e^{\log\pi_{\mathrm{ref}}-\log\pi_\theta}
-(\log\pi_{\mathrm{ref}}-\log\pi_\theta)-1.
$$

The trainer averages valid completion tokens and rollouts, then adds the MoE
router auxiliary loss. One rollout is reused `updates_per_rollout` times:
`old_logps` are cached on update one and stay fixed for later PPO-style updates.

## 2. Dataset

`data/rlaif.jsonl` holds conversations whose final assistant content is empty.
Everything before that final turn is the rollout prompt:

```json
{
  "conversations": [
    {"role": "user", "content": "基于以上对话提出一个问题。"},
    {"role": "assistant", "content": "这些智能家居产品需要哪些前提条件才能够使用？"},
    {"role": "user", "content": "请回答这个问题。"},
    {"role": "assistant", "content": ""}
  ]
}
```

`RLAIFDataset` serializes the prompt with the SFT chat template and appends the
assistant generation header. With `open_thinking: true`, the response begins
with `<think>\n`.

## 3. Prerequisites

Install the environment and download the local reward model once:

```bash
uv sync --extra dev
uv run hf download Skywork/Skywork-Reward-V2-Qwen3-0.6B \
  --local-dir reward_model/Skywork-Reward-V2-Qwen3-0.6B
```

The reward adapter uses `AutoModelForSequenceClassification`, removes system
messages only for reward scoring as required by the model card, and reads the
single Bradley--Terry logit. GRPO starts from the matching SFT checkpoints:

```text
output/sft/dense/step_00100000.pt
output/sft/moe/step_00159670.pt
```

For fast dense rollout on a second GPU, also export the dense SFT checkpoint:

```bash
uv run python -m ajllm.workflows.export_vllm \
  --checkpoint output/sft/dense/step_00100000.pt \
  --tokenizer assets/tokenizers/minimind \
  --output output/vllm_policy/dense
```

Set `rollout_backend: vllm` and the `vllm.model_id` mapping in a copied dense
YAML to use that export. MoE has no vLLM export adapter yet.

## 4. Train

```bash
# Dense: output/grpo/dense/step_00009752.pt
uv run python -m ajllm.workflows.grpo --config configs/grpo/dense.yaml

# Top-1 MoE: output/grpo/moe/step_<latest>.pt after its run completes
uv run python -m ajllm.workflows.grpo --config configs/grpo/moe.yaml
```

Keep `sft_checkpoint` and `reward_model.path` unchanged when resuming. For a
first online smoke run, set `max_steps: 10`; each optimizer update is one of the
`updates_per_rollout` updates over a sampled rollout group.

## 5. Plot

```bash
uv run python -m ajllm.workflows.plot_training --run output/grpo/dense --smooth 20
uv run python -m ajllm.workflows.plot_training --run output/grpo/moe --smooth 20
```

GRPO is detected automatically and produces one PNG with `total_loss`, `reward`,
and `kl` versus optimizer step. Monitor `group_reward_std` in `metrics.jsonl`:
a zero value means that prompt's group has no relative-learning signal.

## 6. Generate

```bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/grpo/dense/step_00009752.pt \
  --messages-json assets/prompt/chat_messages.json --open-thinking

# Replace <latest> with the MoE GRPO checkpoint produced by the run.
uv run python -m ajllm.workflows.generate \
  --checkpoint output/grpo/moe/step_<latest>.pt \
  --messages-json assets/prompt/chat_messages.json --open-thinking

uv run python -m ajllm.workflows.generate \
  --checkpoint output/grpo/dense/step_00009752.pt \
  --prompt-file assets/prompt/continuation.txt
```

## 7. Multi-GPU and vLLM rollout

The supported fast online topology is two GPUs, but it is not distributed GRPO:
GPU 0 holds the trainable policy, reference, and reward model; GPU 1 runs dense
vLLM rollout. `rollout_backend: vllm` deliberately rejects a distributed
policy-training process so the NCCL weight-transfer group remains unambiguous.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run python -m ajllm.workflows.grpo --config configs/grpo/dense.yaml
```

The server receives exact prompt token IDs, synchronizes the current dense
policy before every new rollout group, and intentionally does not synchronize
between repeated updates of that group. Use `rollout_backend: torch` for the
single-GPU dense or MoE path. The broader FSDP/TP/EP topology rules are in
[parallel training](parallel_training.md), but no distributed vLLM GRPO launch
is supported at present.
