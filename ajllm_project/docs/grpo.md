# Group Relative Policy Optimization (GRPO)

GRPO is ajLLM's online RLAIF stage after SFT.  For every prompt ($x_i$), the
current policy samples a group of (G) responses ($y_{i,1}, \ldots, y_{i,G}$).
The frozen local Skywork reward model scores those complete conversations; GRPO
normalizes rewards inside that one prompt's group instead of training a critic.
The policy and frozen SFT reference use ajLLM, while the reward model is only
used for inference.

## Download the reward model

The supplied configurations expect this exact local directory:

~~~bash
uv run hf download Skywork/Skywork-Reward-V2-Qwen3-0.6B \
  --local-dir reward_model/Skywork-Reward-V2-Qwen3-0.6B
~~~

`transformers` and the `hf` CLI are project dependencies.  The model's
`config.json` should identify `Qwen3ForSequenceClassification` and contain a
single label.  GRPO loads it with `AutoModelForSequenceClassification`, applies
its own Qwen chat template, and reads `logits[:, 0]` as the raw
Bradley--Terry reward.  This is deliberately different from MiniMind's
InternLM-specific `get_score()` API.

Skywork's model card recommends no system message when applying its chat
template and reward inference within 16,384 tokens.  The adapter removes system
messages only for reward scoring; ajLLM policy rollouts preserve the full
MiniMind conversation.  The default `reward_model.max_length: 4096` is within
that bound.  See the [official Skywork model card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
for the original sequence-classification loading example and model details.

## Data and rollout

`data/rlaif.jsonl` contains MiniMind conversations ending in an empty assistant
turn.  The final turn is the response slot to be generated; all preceding
messages become the rollout prompt.  `RLAIFDataset` uses the existing SFT chat
serializer, then appends the same assistant header used by chat generation.
With `open_thinking: true`, it also begins the completion with `<think>\n`.

Variable-length trajectories are right-padded for policy and reference logp
evaluation.  Causal attention at every valid token can only see earlier valid
tokens, so padding to the right never becomes context for an action; the loss
mask excludes every padded completion position.  This permits one policy and
one reference forward pass for all `batch_size × num_generations` trajectories.
Set `max_prompt_len + max_new_tokens` no larger than the model `context_length`.

## Objective and token-level implementation

For ($R_{i,j}$), the Skywork reward of generation ($j$) for prompt ($i$), the
group-relative advantage implemented in `group_relative_advantages` is

$$
A_{i,j} = \frac{R_{i,j} - \mu_i}{\sigma_i + \epsilon_A}, \qquad
\mu_i = \frac{1}{G}\sum_{j=1}^{G} R_{i,j}, \qquad
\sigma_i = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_{i,j}-\mu_i)^2}.
$$

Let ($y_{i,j,t}$) be completion token ($t$), ($\pi_{\mathrm{old}}$) the
policy snapshot that sampled the rollout, and ($\pi_{\mathrm{ref}}$) the
frozen SFT reference.  The code computes, for every generated token,

$$
r_{i,j,t}(\theta) =
\frac{\pi_\theta(y_{i,j,t}\mid x_i,y_{i,j,<t})}
     {\pi_{\mathrm{old}}(y_{i,j,t}\mid x_i,y_{i,j,<t})},
\qquad
D_{\mathrm{KL},i,j,t} =
\exp\!\left(\log \pi_{\mathrm{ref}} - \log \pi_\theta\right)
- \left(\log \pi_{\mathrm{ref}} - \log \pi_\theta\right) - 1.
$$

The per-token GRPO loss in `grpo_token_loss` is

$$
\ell_{i,j,t}(\theta) =
-\min\!\left(
r_{i,j,t}(\theta) A_{i,j},
\operatorname{clip}(r_{i,j,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_{i,j}
\right)
+ \beta D_{\mathrm{KL},i,j,t}.
$$

Finally, `GRPOTrainer` averages it over valid generated tokens and over all
\(B \times G\) rollouts in the optimizer batch:

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta) =
\frac{1}{BG}\sum_{i=1}^{B}\sum_{j=1}^{G}
\frac{1}{|y_{i,j}|}\sum_{t=1}^{|y_{i,j}|}\ell_{i,j,t}(\theta)
+ \mathcal{L}_{\mathrm{MoE\ aux}}.
$$

The first batched policy forward after a rollout supplies the cached
\(\log\pi_{\mathrm{old}}\); it is therefore exactly equal to
\(\log\pi_\theta\) before update one.  The same fixed trajectories are then
optimized `updates_per_rollout` times.  Subsequent updates change
\(\pi_\theta\) while retaining the cached \(\pi_{\mathrm{old}}\), making the
importance ratio nontrivial without recomputing rollout log-probabilities.
The reference remains fixed throughout.  For MoE, the same router auxiliary
loss used in pretrain, SFT, and DPO is averaged over rollout sequences and
added to the policy loss.

## Fast rollout backends

`rollout_backend: torch` is a batched fallback: at every decode position it
runs one forward pass for all `batch_size × num_generations` active sequences,
rather than a Python loop per conversation and generation.

For larger runs, set `rollout_backend: vllm` and fill in the commented `vllm`
mapping in the GRPO YAML.  The server receives every prompt in one completion
request with `n: num_generations`; immediately before each *new* rollout group,
the trainer uses `utils/vllm_util.py`'s NCCL weight-transfer path to sync the
current policy.  It intentionally does not sync between the repeated updates
for that group, because those updates must continue to compare against the
fixed behavior policy.

### Dense vLLM export

The dense ajLLM decoder is compatible with the bias-free Qwen3 decoder shape,
including GQA, Q/K RMSNorm, and tied embeddings.  Before enabling vLLM, export
the completed dense SFT checkpoint once:

~~~bash
uv run python -m ajllm.workflows.export_vllm \
  --checkpoint output/sft/dense/step_00100000.pt \
  --tokenizer assets/tokenizers/minimind \
  --output output/vllm_policy/dense
~~~

This creates a standalone HF directory containing `config.json`,
`model.safetensors`, the exact MiniMind tokenizer, and `ajllm_export.json`.
It deliberately refuses to overwrite a non-empty directory.  The exporter is
dense-only: the project MoE layout has custom grouped expert tensors and is not
served by this path.

ajLLM uses interleaved RoPE pairs while Qwen3 uses split-half pairs.  The
exporter reorders Q/K projection rows and Q/K norm gains to make the two
representations mathematically equivalent.  Crucially, the NCCL refresh path
uses the same mapping, so every new rollout uses the current GRPO policy rather
than only the initial exported SFT weights.  vLLM requests receive exact
`prompt` token-ID lists with `add_special_tokens: false`; prompts are never
decoded and re-tokenized between the policy and serving process.

The initial CPU check is included in the repository and compares logits from a
small ajLLM model and its exported Qwen3 model:

~~~bash
uv run --extra dev pytest tests/test_vllm_export.py
~~~

On a two-GPU server, reserve physical GPU 0 for the policy, reference, and
reward model, and physical GPU 1 for vLLM.  This is *not* vLLM tensor
parallelism: vLLM must use one GPU because the other GPU is needed for GRPO
training.  Set the YAML mapping below and run with both devices visible:

~~~yaml
rollout_backend: vllm
vllm:
  model_id: output/vllm_policy/dense
  gpu: 1
  port: 8000
  gpu_memory_utilization: 0.9
~~~

~~~bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run python -m ajllm.workflows.grpo --config configs/grpo/dense.yaml
~~~

`VLLMServer` makes the child process visible only to its configured physical
GPU, while the training process stays on its `cuda:0` (physical GPU 0).  To
test the serving copy independently before GRPO, run the following on GPU 1;
the list under `prompt` is intentionally token IDs, not decoded text:

~~~bash
CUDA_VISIBLE_DEVICES=1 vllm serve output/vllm_policy/dense \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --weight-transfer-config '{"backend":"nccl"}'

curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"output/vllm_policy/dense","prompt":[[1,2,3]],"add_special_tokens":false,"temperature":0,"max_tokens":8,"n":1,"return_token_ids":true}'
~~~

For the full NCCL handoff check, first set `max_steps: 1`,
`updates_per_rollout: 1`, and `rollout_backend: vllm` in a temporary copy of
the dense GRPO config.  A successful run proves the sequence: start server,
initialize the NCCL group, refresh Qwen3-mapped policy weights, generate token
IDs, score them, and apply one policy update.  Do this before a long run.  Do
not manually start the server for this check unless `vllm.launch_server: false`
is also set in the YAML.

## Launch and resume

~~~bash
# Start with max_steps: 10 in YAML; each rollout is reused four policy updates.
uv run python -m ajllm.workflows.grpo --config configs/grpo/dense.yaml

uv run python -m ajllm.workflows.grpo --config configs/grpo/moe.yaml
~~~

`sft_checkpoint` is mandatory even on resume: it restores the immutable
reference policy.  `resume_from` restores the trainable policy, optimizer, and
data position from a GRPO checkpoint.  The reward model is never checkpointed;
its local path and max length are stored in resolved configuration and checkpoint
metadata.  Keep them unchanged when resuming so that rewards retain their
meaning.

Metrics include `reward`, `group_reward_std`, `advantage_std`, `kl`,
`policy_loss`, `auxiliary_loss`, and `response_length`.  A nonzero
`group_reward_std` is important: when all generations for a prompt receive the
same reward, their group-relative advantages are zero and that group produces no
policy-learning signal.

## Generation after GRPO

GRPO checkpoints use the same portable checkpoint structure as SFT and DPO, so
existing generation needs no change:

~~~bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/grpo/dense/step_00000500.pt \
  --messages-json messages.json --open-thinking
~~~
