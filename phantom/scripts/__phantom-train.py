#!/bin/python3
"""
Script for performing training of Phantom RL experiments.

Usage
    python train.py config_file

"""
import argparse
import sys

from phantom.cmd_utils import train_from_config_path


def main(args):
    parser = argparse.ArgumentParser(
        prog="phantom-train",
        description="Runs RL training for an experiment defined in the given Phantom experiment config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config_file", type=str, help="Name of the experiment configuration file."
    )

    flags = parser.parse_known_args(args)[0]

    print()
    print("╔═══════════════╗")
    print("║ Phantom-Train ║")
    print("╚═══════════════╝")
    print()

    results_dir = train_from_config_path(flags.config_file)

    print("\n" + ("═" * 80))

    if results_dir is not None:
        print(f"\nResults saved to: {results_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main(sys.argv[1:])
