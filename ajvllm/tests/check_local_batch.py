"""Small, opt-in real-checkpoint check: one FP32 model, <= 8 context tokens.

Run separately: uv run python tests/check_local_batch.py
No second oracle model is loaded. Tiny CUDA tests provide independent HF parity.
"""

import json
from pathlib import Path

import torch
from model_inputs import forward_tokens

from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput


def main():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) < 4 * 1024**2:
        raise SystemExit("At least 4 GiB of available host memory is required for this optional check")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.2)
    torch.backends.cuda.matmul.allow_tf32 = False
    runner = Qwen2Runner.from_directory("model/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
    sequences = [(1, 2, 3, 4, 5, 6, 7, 8), (2, 3, 4, 5), (3, 4, 5, 6, 7, 8)]
    with torch.inference_mode():
        runner._caches["2"] = forward_tokens(runner.model, torch.tensor(sequences[2][:-1], device="cuda")).caches[0]
        plan = SchedulerOutput(
            (
                ScheduledRequest("0", sequences[0], 0, Phase.PREFILL, True),
                ScheduledRequest("1", sequences[1], 0, Phase.PREFILL, True),
                ScheduledRequest("2", sequences[2][-1:], 5, Phase.DECODE, True),
            )
        )
        actual = runner.execute(plan)
        errors = []
        for row, sequence in enumerate(sequences):
            expected = forward_tokens(runner.model, torch.tensor(sequence, device="cuda"), logits_to_keep=1).logits[-1]
            logits = actual[str(row)]
            torch.testing.assert_close(logits, expected, atol=2e-4, rtol=2e-5)
            errors.append((logits - expected).abs().max().item())
            runner.release(str(row))
        assert runner.cache_bytes == 0
    print(
        json.dumps(
            {
                "max_errors": errors,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "batch_size": len(plan.requests),
                "remaining_cache_bytes": runner.cache_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
