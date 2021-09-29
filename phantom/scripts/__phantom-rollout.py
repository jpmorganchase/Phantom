#!/bin/python3
"""
Script for performing rollouts of Phantom RL experiments.

For instructions, use:
    phantom-rollout --help

"""
import argparse
import sys

import coloredlogs

from phantom.params import RolloutParams
from phantom.utils import load_object
from phantom.utils.rollout import run_rollouts


def main(args):
    coloredlogs.install(
        level="INFO",
        fmt="(pid=%(process)d) %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="phantom-rollout",
        description="Performs rollouts for an already trained experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config_file", type=str, help="The Python config file defining the experiment."
    )

    flags = parser.parse_known_args(args)[0]

    print()
    print("╔═════════════════╗")
    print("║ Phantom-Rollout ║")
    print("╚═════════════════╝")
    print()

    params = load_object(flags.config_file, "rollout_params", RolloutParams)

    run_rollouts(params)


if __name__ == "__main__":
    main(sys.argv[1:])
