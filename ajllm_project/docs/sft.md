# Supervised fine-tuning

SFT starts from a portable ajLLM pre-training checkpoint and trains only on
assistant completions in MiniMind conversational JSONL. Each record has the
following shape:

~~~json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
~~~

System messages, assistant reasoning_content, tool declarations, assistant tool
calls, and tool response messages are also supported. The dataset reproduces the
bundled MiniMind chat layout (im_start role markers, think blocks, tool calls,
and im_end markers). Prompt tokens, tool-response tokens, and the assistant role
header have label -100; the assistant completion, its thinking block, tool-call
payload, and ending token are the only optimized targets. Long dialogues retain
their most recent max_seq_len + 1 tokens so the final completion is not dropped.

The configuration tree separates model architecture from stage-specific runs:

~~~text
configs/model/{dense,moe}.yaml
configs/pretrain/{dense,dense_tp4,moe,moe_ep4,moe_tp2_ep4}.yaml
configs/sft/{dense,dense_tp4,dense_fsdp2,moe,moe_ep4,moe_tp2_ep4}.yaml
~~~

## Single RTX 3080 Ti

Start with the supplied BF16 configuration when your CUDA/PyTorch installation
reports BF16 support; otherwise change mixed_precision from bf16 to fp16. First
run a short job:

~~~bash
# dense
uv run python -m ajllm.workflows.sft --config configs/sft/dense.yaml

# moe
uv run python -m ajllm.workflows.sft --config configs/sft/moe.yaml
~~~

Set max_steps to 100 for the first smoke run. pretrained_checkpoint must point
to the completed pre-training checkpoint with the exact same model configuration
and tokenizer. It initializes weights only; it deliberately does not restore a
pre-training optimizer state. To continue an interrupted SFT run, set
pretrained_checkpoint to null and set resume_from to its SFT checkpoint.
Older Top-1 MoE pre-training checkpoints that store separate expert gate, up,
and down projections are converted automatically into the current grouped-expert
layout during SFT initialization.

## Multi-GPU

The SFT entry point uses exactly the same FSDP, TP, EP, TP×EP, sampler, metric,
and portable-checkpoint implementation as pre-training. The included dense
examples are:

~~~bash
# 4 GPUs: dense tensor-parallel SFT
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.sft --config configs/sft/dense_tp4.yaml

# 2 GPUs: FULL_SHARD data-parallel SFT
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=2 \
  -m ajllm.workflows.sft --config configs/sft/dense_fsdp2.yaml

# 4 GPUs: MoE expert-parallel SFT
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.sft --config configs/sft/moe_ep4.yaml

# 8 GPUs: MoE tensor + expert parallel SFT
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=8 \
  -m ajllm.workflows.sft --config configs/sft/moe_tp2_ep4.yaml
~~~

For single-GPU MoE SFT use configs/sft/moe.yaml. It and the multi-GPU examples
use configs/model/moe.yaml plus a matching MoE pre-training checkpoint.
TP ranks consume the same examples; EP and FSDP ranks consume separate examples.
Keep every .rank<N>.optim file beside a parallel SFT checkpoint when resuming it.

## Chat inference and tools

Plain generate --prompt remains available for pre-training text prompts. For an
SFT conversation, pass a JSON message array to --messages-json; this uses the
same MiniMind serialization as the SFT dataset and returns structured assistant
content, optional reasoning_content, and optional tool_calls.

~~~bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/sft/moe/step_00000100.pt \
  --messages-json messages.json --open-thinking
~~~

Tool schemas can be supplied with --tools-json to let the model decide whether
to issue a call. Actual local tool execution is intentionally a Python API
operation: pass a name-to-callable registry to generate_chat. The loop accepts
only registered names, appends each JSON tool response to the conversation, then
calls the model again, up to max_tool_rounds.

~~~python
from ajllm.workflows.generate import generate_chat

result = generate_chat(
    model, tokenizer, [{"role": "user", "content": "Double 21."}],
    max_new_tokens=128, temperature=0.0, top_k=0, top_p=1.0, device=device,
    tools=[{"function": {"name": "double", "parameters": {"type": "object"}}}],
    tool_registry={"double": lambda value: {"result": value * 2}},
)
~~~

The model is never allowed to turn a generated function name into a shell
command, import, or unrestricted callable.
