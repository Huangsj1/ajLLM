#!/usr/bin/env python3
"""Launch distributed training with torchrun.

This script wraps torchrun to simplify launching multi-GPU training jobs.

Examples:
    # 2 GPUs on single node
    python scripts/launch_distributed.py \\
        --config configs/runs/lm_train/tinystories_baseline.yaml \\
        --nproc-per-node 2

    # 4 GPUs on single node with custom port
    python scripts/launch_distributed.py \\
        --config configs/runs/lm_train/openwebtext_baseline.yaml \\
        --nproc-per-node 4 \\
        --master-port 29500
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--nproc-per-node", type=int, default=2, help="Number of processes per node (GPUs)")
    parser.add_argument("--master-port", type=str, default="29500", help="Master port for distributed coordination")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes (for multi-node training)")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master node address")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node-rank={args.node_rank}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        "-m", "ajllm",
        "lm", "train",
        "--config", str(config_path),
    ]

    print(f"Launching distributed training with {args.nproc_per_node} GPUs...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Execute torchrun
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
