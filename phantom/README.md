# Phantom

> A Multi-agent reinforcement-learning simulator built on top of RLlib.


## Installation


```sh
pip install -r requirements.txt
python setup.py install
```


## Architecture

Please see the Phantom white-paper for an in-depth guide to the Phantom architecture and
design principles.

## Usage

Included in Phantom are several commands that make it simple to run and analyse experiments.

For each command, passing the `--help` flag will show full instructions and list any
additional available arguments.

### phantom-train

The `phantom-train` command takes a Phantom-style Python configuration file as an input
and performs experiment training:

```sh
phantom-train path/to/config.py
```

### phantom-rollout

The `phantom-rollout` commands takes a Phantom-style Python configuration file and a path
to a directory containing results for that experiment as inputs and performs rollouts on
the results:

```sh
phantom-rollout path/to/config.py ~/phantom_results/experiment_name/...
```

### phantom-profile

The `phantom-profile` command is similar to the `phantom-train` command with the difference
being that it runs a performance profiler over the experiment and outputs the results of
this as a compute graph. Note: for this to be performed sucessfully the command will
setup the experiment to use only one CPU.

```sh
phantom-profile path/to/config.py
```


## Usage example

TODO


Further examples can be found in the examples directory. These can be run with the included
command, for example:

```sh
phantom-train examples/monopoly.py
```


## Reproducibility

ML experiment reproducibility can be hard. There are several common pitfalls in Python
that can be hard to spot. To reduce these problems it is recommended that the user uses
the included `phantom-train` and `phantom-rollout` commands as these are configured to
produce reproducible results.


## Development setup


```sh
make install_deps
make dev
```
