#!/bin/python3
"""
Script for performing rollouts of Phantom RL experiments.

For instructions, use:
    phantom-rollout --help

"""
import argparse
import sys
from pathlib import Path

from phantom.cmd_utils import load_config, rollout


def main(args):
    parser = argparse.ArgumentParser(
        prog="phantom-rollout",
        description="Performs rollouts for an already trained experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config_file", type=str, help="The Python config file defining the experiment."
    )
    parser.add_argument(
        "results_dir", type=str, help="Ray results directory of the experiment."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="How many Ray workers to use (default = 1).",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="The number of rollouts to perform (default = 1).",
    )
    parser.add_argument(
        "--checkpoint-num",
        type=int,
        help="The checkpoint to use (defaults to most recent checkpoint).",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics.pkl",
        help="The file to save the rollout metrics to (in pickle format).",
    )
    parser.add_argument(
        "--replays-file",
        type=str,
        help="The file to save the rollout replays to (in pickle format).",
    )

    flags = parser.parse_known_args(args)[0]

    # If the checkpoint num flag is not set, find the largest checkpoint num in
    # the results directory.
    if flags.checkpoint_num is None:
        checkpoint_dirs = sorted(Path(flags.results_dir).glob("checkpoint_*"))
        flags.checkpoint_num = int(str(checkpoint_dirs[-1]).split("_")[-1])

    print()
    print("╔═════════════════╗")
    print("║ Phantom-Rollout ║")
    print("╚═════════════════╝")
    print()
    print(f"Experiment   : {flags.results_dir}")
    print(f"Checkpoint   : {flags.checkpoint_num}")
    print(f"Num Rollouts : {flags.num_rollouts}")
    print(f"Num Workers  : {flags.num_workers}")
    print(f"Metrics File : {flags.metrics_file}")
    print(f"Replays File : {flags.replays_file}")
    print()

    _, config_params = load_config(flags.config_file)

    results = rollout(
        config_params,
        flags.results_dir,
        flags.checkpoint_num,
        flags.num_rollouts,
        flags.num_workers,
        flags.metrics_file,
        flags.replays_file,
    )

    if results is not None:
        print()

        if flags.metrics_file is not None:
            metrics_path = Path(flags.results_dir, flags.metrics_file)
            print(f"Metrics saved to {metrics_path}")

        if flags.replays_file is not None:
            replays_path = Path(flags.results_dir, flags.replays_file)
            print(f"Replays saved to {replays_path}")

        print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
