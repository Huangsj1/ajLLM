"""Rebuild deterministic long English prompts using only the local tokenizer."""

import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("model/Qwen2.5-0.5B-Instruct", local_files_only=True)
    rows = []
    topics = ["community library", "urban garden", "regional railway", "public museum"]
    for target in (1024, 2048, 3072):
        for index, topic in enumerate(topics):
            introduction = f"Read the following operational reports about a {topic}.\n\n"
            sections = []
            for week in range(1, 100):
                sections.append(
                    f"Week {week}: The {topic} recorded {120 + week * (index + 3)} visitors. "
                    f"The team had {4 + week % 5} volunteers and a budget of {300 + week * 7} dollars. "
                    "Morning demand was lower than afternoon demand. Staff recorded waiting times, "
                    "maintenance requests, and feedback in a shared notebook. A delayed delivery "
                    "required the team to postpone one activity, while a new sign improved access "
                    "for first-time visitors. The coordinator proposed a rotating schedule to spread "
                    "work fairly. Members disagreed about whether limited funds should support "
                    "new equipment or repairs. They agreed to review costs and accessibility before "
                    "making a decision. The next report should compare outcomes with the previous "
                    "week and identify unresolved risks.\n\n"
                )
            ending = (
                "\n\nSummarize the recurring problems and propose three practical improvements, "
                "explaining the tradeoffs."
            )
            prefix = tokenizer.encode(introduction + "".join(sections), add_special_tokens=False)
            suffix = tokenizer.encode(ending, add_special_tokens=False)
            prompt = tokenizer.decode(prefix[: target - len(suffix)]) + ending
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
            assert abs(count - target) <= 2 and count + 64 <= 4096
            rows.append({"id": f"{topic.replace(' ', '-')}-{target}", "prompt": prompt, "prompt_tokens": count})
    output = Path("benchmarks/datasets/long.jsonl")
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"{output}: {len(rows)} prompts; token lengths {[row['prompt_tokens'] for row in rows]}")


if __name__ == "__main__":
    main()
