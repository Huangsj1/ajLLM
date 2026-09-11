# Direct Preference Optimization (DPO)

DPO is the preference-alignment stage after SFT.  It starts with two identical
copies of a completed SFT checkpoint: a trainable policy ($\pi_\theta$), and a
frozen reference policy ($\pi_{\mathrm{ref}}$).  For each prompt ($x$), the
dataset supplies a preferred completion ($y_w$) (`chosen`) and a weaker
completion ($y_l$) (`rejected`).  No reward model or on-policy rollout is
needed.

The implemented objective is:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) =
-\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[
\log \sigma \left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
\right].
$$

Equivalently, the trainer sums token log-probabilities over each assistant
completion, computes the policy/reference log-ratio for `chosen` and
`rejected`, subtracts the latter from the former, multiplies by `beta`, and
applies `-logsigmoid`.  Router auxiliary loss is added for MoE models, exactly
as in pre-training and SFT.

## Dataset

`data/dpo.jsonl` uses the MiniMind preference-pair schema:

~~~json
{
  "chosen": [
    {"role": "user", "content": "Explain DPO."},
    {"role": "assistant", "content": "...preferred answer..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain DPO."},
    {"role": "assistant", "content": "...weaker answer..."}
  ]
}
~~~

The two arrays must have an identical prefix and each must end with an
assistant message.  This makes the shared prefix the DPO prompt.  Messages are
serialized with the same MiniMind template as SFT and `generate_chat`; system
messages, thinking blocks, tools, and tool calls therefore retain the same
format.  Only assistant completion tokens contribute to sequence
log-probabilities.  Each branch keeps its final `max_seq_len + 1` serialized
tokens, just like SFT.

## Launch

The supplied dense and MoE configurations point at the completed SFT
checkpoints currently produced by this project:

~~~bash
# Set max_steps: 100 in the selected YAML for an initial smoke run.
uv run python -m ajllm.workflows.dpo --config configs/dpo/dense.yaml
uv run python -m ajllm.workflows.dpo --config configs/dpo/moe.yaml
~~~

`sft_checkpoint` is mandatory, even when resuming: it is the immutable
reference model.  `resume_from` restores only the trainable DPO policy,
optimizer, sampler position, and step counter.  Do not replace
`sft_checkpoint` for a resumed run; doing so changes the objective.  The DPO
checkpoint metadata records the reference path and has the same portable model
format as every other stage.

DPO holds a policy and a reference model concurrently, so reduce `batch_size`
or increase gradient accumulation before reducing `max_seq_len` if VRAM is
tight.  The same FSDP, TP, EP, and TP×EP wrappers used by the prior stages are
supported through the standard `use_fsdp` / `parallel` YAML fields; each rank
also owns a frozen reference-model shard.

The metrics JSONL includes `dpo_loss`, `auxiliary_loss`, `chosen_reward`,
`rejected_reward`, `reward_margin`, and `preference_accuracy`.  A positive
margin and an accuracy above 0.5 indicate that the policy has moved toward the
preferred responses relative to the frozen reference.

## Generation after DPO

No special generation implementation is needed.  The existing portable
checkpoint loader reads DPO checkpoint metadata, so use the standard chat
generation command with the resulting policy checkpoint:

~~~bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/dpo/dense/step_00005000.pt \
  --messages-json messages.json --open-thinking
~~~

For plain continuation prompts, replace `--messages-json ...` with
`--prompt "..."`.  Tool handling and the registered-callable restriction are
unchanged from SFT generation.
