<div id="top"></div>

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0 License][license-shield]][license-url] -->

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
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#license">License</a></li>
    <!-- <li><a href="#contact">Contact</a></li> -->
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
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
(3.6 minimum) and access to the pip Python package manager.

A list of Python packages required by Phantom/Mercury is given in the
``requirements.txt`` files in each respective directory. The required packages can be
installed by running::

    make install_deps

### Phantom

Phantom and its dependency Mercury can be installed as libraries with the command::

    make install


### Mercury

Mercury can be installed independently as a library with the command::

    make install-mercury

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

With Phantom installed you can run the provided ``supply-chain`` sample experiment
with the command::

    phantom envs/supply-chain/supply-chain-1.py

Change the script for any of the other provided experiments in the examples directory.


<p align="right">(<a href="#top">back to top</a>)</p>


## Contributing

TODO: add terms of CLA


<p align="right">(<a href="#top">back to top</a>)</p>


## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p> -->


<!-- ## Acknowledgments


<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- [contributors-shield]: https://img.shields.io/github/contributors/jpmorganchase/Phantom.svg?style=for-the-badge
[contributors-url]: https://github.com/jpmorganchase/Phantom/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/jpmorganchase/Phantom.svg?style=for-the-badge
[forks-url]: https://github.com/jpmorganchase/Phantom/network/members

[stars-shield]: https://img.shields.io/github/stars/jpmorganchase/Phantom.svg?style=for-the-badge
[stars-url]: https://github.com/jpmorganchase/Phantom/stargazers

[issues-shield]: https://img.shields.io/github/issues/jpmorganchase/Phantom.svg?style=for-the-badge
[issues-url]: https://github.com/jpmorganchase/Phantom/issues

[license-shield]: https://img.shields.io/github/license/jpmorganchase/Phantom.svg?style=for-the-badge
[license-url]: https://github.com/jpmorganchase/Phantom/blob/master/LICENSE.txt

[product-screenshot]: images/screenshot.png -->


<!-- README template used: https://github.com/othneildrew/Best-README-Template -->