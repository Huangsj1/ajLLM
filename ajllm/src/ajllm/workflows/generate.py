"""Plain-text and MiniMind-chat generation from portable checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ajllm.datasets import encode_sft_conversation
from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer
from ajllm.workflows.causal_lm import _upgrade_legacy_top1_moe_state_dict

ToolRegistry = Mapping[str, Callable[..., Any]]
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_PATTERN = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class ChatGeneration:
    """Final assistant turn and the complete conversation after tool rounds."""

    message: dict[str, Any]
    messages: list[dict[str, Any]]
    tool_rounds: int


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    shifted = logits - torch.amax(logits)
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Sample one ID with primitive tensor operations and optional filtering."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    filtered = logits.float() / temperature
    if top_k > 0:
        cutoff = torch.topk(filtered, min(top_k, filtered.numel())).values[-1]
        filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probabilities = _softmax(sorted_logits)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = filtered.scatter(0, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    probabilities = _softmax(filtered)
    return int(torch.multinomial(probabilities, 1).item())


@torch.no_grad()
def _generate_ids(
    model: torch.nn.Module,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    eos_token_id: int,
) -> list[int]:
    """Generate from already-rendered IDs, retaining only model context length."""
    context_length = model.config.context_length
    context = token_ids[-context_length:]
    generated_ids: list[int] = []
    model.eval()
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        token_id = _sample(model(input_ids)[0, -1], temperature, top_k, top_p)
        if token_id == eos_token_id:
            break
        generated_ids.append(token_id)
        context = [*context, token_id][-context_length:]
    return generated_ids


def generate(
    model: torch.nn.Module,
    tokenizer: MiniMindTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    """Generate a plain continuation for a pre-training-style text prompt."""
    token_ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt)]
    generated_ids = _generate_ids(
        model, token_ids, max_new_tokens, temperature, top_k, top_p, device, tokenizer.eos_token_id
    )
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _with_tools(messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None) -> list[dict[str, Any]]:
    history = [dict(message) for message in messages]
    if not history:
        raise ValueError("messages must contain at least one user or system message")
    if tools is None:
        return history
    if history[0].get("role") == "system":
        history[0]["tools"] = list(tools)
    else:
        history.insert(0, {"role": "system", "content": "", "tools": list(tools)})
    return history


def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    tool_calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            tool_calls.append({"function": {"name": payload["name"], "arguments": payload.get("arguments", {})}})
    return _TOOL_CALL_PATTERN.sub("", text).strip(), tool_calls


def parse_assistant_message(text: str) -> dict[str, Any]:
    """Parse a MiniMind assistant completion into OpenAI-style message fields."""
    content_with_think, tool_calls = _parse_tool_calls(text)
    reasoning_content = None
    think_match = _THINK_PATTERN.match(content_with_think)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        content_with_think = content_with_think[think_match.end() :]
    message: dict[str, Any] = {"role": "assistant", "content": content_with_think.strip()}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_response(tool_call: Mapping[str, Any], tool_registry: ToolRegistry) -> str:
    function = tool_call.get("function", tool_call)
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        return json.dumps({"error": "invalid_tool_call"}, ensure_ascii=False)
    name = function["name"]
    arguments = function.get("arguments", {})
    try:
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(parsed_arguments, dict):
            raise TypeError("tool arguments must be a JSON object")
        handler = tool_registry[name]
        result = handler(**parsed_arguments)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error), "tool": name}, ensure_ascii=False)


def generate_chat(
    model: torch.nn.Module,
    tokenizer: MiniMindTokenizer,
    messages: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    *,
    tools: Sequence[Any] | None = None,
    tool_registry: ToolRegistry | None = None,
    max_tool_rounds: int = 4,
    open_thinking: bool = False,
) -> ChatGeneration:
    """Generate a MiniMind chat completion, optionally executing registered tools.

    Tool execution is opt-in. A model-provided function name can only invoke a
    callable explicitly present in tool_registry; no shell commands, imports,
    or arbitrary code are accepted from model output.
    """
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be non-negative")
    history = _with_tools(messages, tools)
    prompt_suffix = (
        f"{tokenizer.bos_token}assistant\n<think>\n"
        if open_thinking
        else f"{tokenizer.bos_token}assistant\n<think>\n\n</think>\n\n"
    )
    for tool_round in range(max_tool_rounds + 1):
        token_ids, _ = encode_sft_conversation(tokenizer, history)
        token_ids.extend(tokenizer.encode(prompt_suffix))
        generated_ids = _generate_ids(
            model, token_ids, max_new_tokens, temperature, top_k, top_p, device, tokenizer.eos_token_id
        )
        assistant_message = parse_assistant_message(tokenizer.decode(generated_ids, skip_special_tokens=False))
        history.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls or tool_registry is None or tool_round == max_tool_rounds:
            return ChatGeneration(assistant_message, history, tool_round)
        history.extend(
            {"role": "tool", "content": _tool_response(tool_call, tool_registry)} for tool_call in tool_calls
        )
    raise AssertionError("tool loop must return within max_tool_rounds")


def _load_json(path: str | Path, description: str) -> Any:
    with Path(path).open(encoding="utf-8") as source:
        try:
            return json.load(source)
        except json.JSONDecodeError as error:
            raise ValueError(f"{description} must contain valid JSON") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plain text or MiniMind SFT chat completions")
    parser.add_argument("--checkpoint", required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Plain pre-training-style prompt")
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing a plain continuation prompt")
    prompt_group.add_argument("--messages-json", help="JSON file containing a conversation message array")
    parser.add_argument("--tools-json", help="Optional JSON file containing tool schemas for chat prompting")
    parser.add_argument("--open-thinking", action="store_true", help="Let the assistant generate its think block")
    parser.add_argument("--tokenizer", default="assets/tokenizers/minimind")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["metadata"]["model_config"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = build_model(model_config).to(device)
    model.load_state_dict(_upgrade_legacy_top1_moe_state_dict(checkpoint["model_state_dict"], model))
    torch.manual_seed(args.seed)
    tokenizer = MiniMindTokenizer.from_pretrained(args.tokenizer)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file is not None else args.prompt
    if prompt is not None:
        print(
            generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, device)
        )
        return
    messages = _load_json(args.messages_json, "messages-json")
    tools = _load_json(args.tools_json, "tools-json") if args.tools_json else None
    if not isinstance(messages, list) or tools is not None and not isinstance(tools, list):
        raise ValueError("messages-json and tools-json must each contain a JSON array")
    result = generate_chat(
        model,
        tokenizer,
        messages,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        device,
        tools=tools,
        open_thinking=args.open_thinking,
    )
    print(json.dumps({"message": result.message, "messages": result.messages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
