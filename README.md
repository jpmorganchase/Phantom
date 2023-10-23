<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0 License][license-shield]][license-url]

<br />
<div align="center">
  <h2 align="center">Phantom</h2>

  <a href="https://www.jpmorgan.com/technology/artificial-intelligence">
    <img src="docs/img/ai.png" alt="JPMorgan AI Research Logo" width=300>
  </a>

  <p align="center">
    A Multi-agent reinforcement-learning simulator framework.
    <br />
    <a href="https://phantom-marl.readthedocs.io"><strong>Explore the docs »</strong></a>
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


## Installation

### Prerequisites

The main requirements for installing Phantom are a modern Python installation
(3.8 minimum) and access to the pip Python package manager.

A list of Python packages required by Phantom is given in the
`requirements.txt` files in each respective directory. The required packages can be
installed by running:

```sh
make install-deps
```

### Phantom

Phantom can be installed as libraries with the command::

```sh
make install
```

To use the network plotting feature for Tensorboard the following additional packages
are required:

- matplotlib
- networkx


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

With Phantom installed you can run the provided `supply-chain` sample experiment
with the command:

```sh
phantom examples/environments/supply-chain/supply-chain.py
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
@inproceedings{ardon2023phantom,
  title={Phantom-A RL-driven Multi-Agent Framework to Model Complex Systems},
  author={Ardon, Leo and Vann, Jared and Garg, Deepeka and Spooner, Thomas and Ganesh, Sumitra},
  booktitle={Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
  pages={2742--2744},
  year={2023},
  keywords = {Artificial Intelligence, Multiagent Systems, Reinforcement Learning}
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
