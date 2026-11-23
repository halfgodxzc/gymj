"""Run four greedy R5 seats using only public decision features."""

import argparse

import torch

from gymj.v4_agent import V4Agent, run_v4_round


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run_v4_round(V4Agent.from_checkpoint(args.actor), args.seed)
    if sum(result.score_deltas) != 0:
        raise RuntimeError("round settlement is not zero-sum")
    print(result.seed, result.step_count, result.end_reason, result.score_deltas)


if __name__ == "__main__":
    main()
