# Usage Guide

See the `README.md` files for instructions on how to install Phantom/Mercury and their
dependencies.


## Reporting Bugs, Feature Requests, etc.

To report bugs, request new features or similar, please [open an issue on the Github repository](https://github.com/jpmorganchase/Phantom/issues).

For reporting bugs an Issue Template ([ISSUE-TEMPLATE.md]()) is provided to help guide
you to what information should be included in the issue.


## Versions

Releases follow the [semver](https://semver.org) versioning scheme:

- `MAJOR.MINOR.PATCH`:
  - `MAJOR` version when incompatible API changes are made,
  - `MINOR` version when functionality is added in a backwards compatible manner, and
  - `PATCH` version when backwards compatible bug fixes are made.

For example:

A future version 1.4.0 of Phantom will be backwards compatible with a future version
1.2.0. You should be able to update your projects dependencies from 1.2.0 to 1.4.0
without any breakages to your code.

A future version 2.0.0 of Phantom may not be backwards compatible with version 1.4.0.
Upgrading to this version may introduce breaking API changes. These will be documented
in the release notes along with instructions on how to update your code. 
