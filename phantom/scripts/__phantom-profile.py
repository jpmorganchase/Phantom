#!/bin/python3
"""
Script for performing training of Phantom RL experiments.

Usage
    python train.py config_file

"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

from profilehooks import profile

from phantom.params import TrainingParams
from phantom.utils import load_object
from phantom.utils.training import train_from_params_object


def main(args):
    parser = argparse.ArgumentParser(
        prog="phantom-profile",
        description="Runs RL training for an experiment and profiles the execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config_file", type=str, help="Name of the experiment configuration file."
    )

    flags = parser.parse_known_args(args)[0]

    print()
    print("╔═════════════════╗")
    print("║ Phantom-Profile ║")
    print("╚═════════════════╝")
    print()

    params = load_object(flags.config_file, "training_params", TrainingParams)

    timestamp_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    filename_root = f"phantom-profiling_{timestamp_str}"

    @profile(stdout=False, immediate=True, filename=f"{filename_root}.prof")
    def run():
        train_from_params_object(params, local_mode=True, config_path=flags.config_file)

    run()

    print("\nSimulation completed. Generating profiling results!")

    subprocess.call(
        f"gprof2dot {filename_root}.prof -f pstats > {filename_root}.dot", shell=True
    )
    subprocess.call(f"dot -Tsvg -o {filename_root}.svg {filename_root}.dot", shell=True)

    os.remove(f"{filename_root}.dot")
    os.remove(f"{filename_root}.prof")

    print(f"Profiling results saved to '{filename_root}.svg'")


if __name__ == "__main__":
    main(sys.argv[1:])
