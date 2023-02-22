<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0 License][license-shield]][license-url]

<br />
<div align="center">
  <h2 align="center">Phantom / Mercury</h2>

  <a href="https://github.com/jpmorganchase/Phantom">
    <img src="phantom/docs/img/ai.png" alt="JPMorgan AI Research Logo" width=300>
  </a>

  <p align="center">
    A Multi-agent reinforcement-learning simulator framework.
    <br />
    <a href="https://github.com/jpmorganchase/Phantom"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/jpmorganchase/Phantom/issues">Report Bug</a>
    ·
    <a href="https://github.com/jpmorganchase/Phantom/issues">Request Feature</a>
  </p>
</div>


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-phantom">About Phantom</a></li>
    <li><a href="#about-mercury">About Mercury</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#citing-phantom">Citing Phantom</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


## About Phantom

Phantom is a multi-agent reinforcement-learning simulator built on top of RLlib.

<p align="right">(<a href="#top">back to top</a>)</p>


## About Mercury

Mercury is a library for simulating P2P messaging networks. It is all built on
`networkx` primitives and enforces strict observability constraints as a first-class
feature.

<p align="right">(<a href="#top">back to top</a>)</p>


## Installation

### Prerequisites

The main requirements for installing Phantom/Mercury are a modern Python installation
(3.8 minimum) and access to the pip Python package manager.

A list of Python packages required by Phantom/Mercury is given in the
`requirements.txt` files in each respective directory. The required packages can be
installed by running:

```sh
make install-deps
```

### Phantom

Phantom and its dependency Mercury can be installed as libraries with the command::

```sh
make install
```

To use the network plotting feature for Tensorboard the following additional packages
are required:

- matplotlib
- networkx


### Mercury

Mercury can be installed independently as a library with the command:

```sh
make install-mercury
```

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

With Phantom installed you can run the provided `supply-chain` sample experiment
with the command:

```sh
phantom envs/supply-chain/supply-chain-1.py
```

Change the script for any of the other provided experiments in the examples directory.

<p align="right">(<a href="#top">back to top</a>)</p>


## Contributing

Thank you for your interest in contributing to Phantom!

We invite you to contribute enhancements. Upon review you will be required to complete
the [Contributor License Agreement (CLA)](https://github.com/jpmorganchase/cla) before
we are able to merge.

If you have any questions about the contribution process, please feel free to send an
email to [open_source@jpmorgan.com](mailto:open_source@jpmorgan.com).

<p align="right">(<a href="#top">back to top</a>)</p>


## Citing Phantom

Find the paper on Arxiv [Phantom -- An RL-driven framework for agent-based modeling of complex economic systems and markets](https://arxiv.org/abs/2210.06012) or use the following BibTeX:

```
@misc{https://doi.org/10.48550/arxiv.2210.06012,
  doi = {10.48550/ARXIV.2210.06012},
  url = {https://arxiv.org/abs/2210.06012},
  author = {Ardon, Leo and Vann, Jared and Garg, Deepeka and Spooner, Tom and Ganesh, Sumitra},
  keywords = {Artificial Intelligence (cs.AI), Multiagent Systems (cs.MA), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Phantom -- An RL-driven framework for agent-based modeling of complex economic systems and markets},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>


## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


[contributors-shield]: https://img.shields.io/github/contributors/jpmorganchase/Phantom.svg?style=for-the-badge
[contributors-url]: https://github.com/jpmorganchase/Phantom/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/jpmorganchase/Phantom.svg?style=for-the-badge
[forks-url]: https://github.com/jpmorganchase/Phantom/network/members

[stars-shield]: https://img.shields.io/github/stars/jpmorganchase/Phantom.svg?style=for-the-badge
[stars-url]: https://github.com/jpmorganchase/Phantom/stargazers

[issues-shield]: https://img.shields.io/github/issues/jpmorganchase/Phantom.svg?style=for-the-badge
[issues-url]: https://github.com/jpmorganchase/Phantom/issues

[license-shield]: https://img.shields.io/github/license/jpmorganchase/Phantom.svg?style=for-the-badge
[license-url]: https://github.com/jpmorganchase/Phantom/blob/master/LICENSE.txt


<!-- README template used: https://github.com/othneildrew/Best-README-Template -->