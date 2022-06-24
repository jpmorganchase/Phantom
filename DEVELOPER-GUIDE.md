# Developer Guide

## Git Policies

The main development branch is the `develop` branch. Actual development of features and
improvements should be done on individual feature branches and then merged into the
`develop` branch via pull requests.

Code from the `develop` branch will be merged into the `main` branch when releases are
tagged and published.

- It is acceptable for some small commits to be applied directly to the `develop` branch.
  Examples include:
  - Spelling/wording corrections.
  - Minor formatting changes.
- Larger commits that have an effect on code and execution must be developed on a
  separate branch and merged with a pull request.
- Pull requests must pass all GitHub Actions [tests](https://github.com/jpmorganchase/Phantom/blob/main/.github/workflows/workflow.yml)
  and must have at least one reviewer.
- Use of GitHub issues and road-maps for tracking improvements and bugs is encouraged.


## Develop Branch Merge Requirements

- Any additions or changes must be well documented.
  - Code must have an appropriate amount of comment strings.
  - Functions, classes and methods must have conforming document strings.
  - Function signatures and class properties must be well typed.
  - New classes and appropriate functions must be added to the Sphinx documentation.
  - Major new or changed functionality must be documented in the "User Guide" section
    of the Sphinx documentation.
- All unit tests must pass.
  - This can be checked locally with the command `make test`.
- Any major new functionality must have unit tests written covering the code.
  - This can be checked locally with the command `make cov`.
- Code should aim to follow the official Python [PEP8](https://www.python.org/dev/peps/pep-0008/)
  style guide.
- Python code must be formatted in accordance with the [black](https://black.readthedocs.io/en/stable/)
  code formatting tool.
  - This can be applied locally with the command `make format`.
- Changes must be added to the rolling `CHANGELOG.md` file.


## Versioning

- Releases should follow the [semver](https://semver.org) versioning scheme:
  - `MAJOR.MINOR.PATCH`:
    - `MAJOR` version when you make incompatible API changes,
    - `MINOR` version when you add functionality in a backwards compatible manner, and
    - `PATCH` version when you make backwards compatible bug fixes.


## Release Process

Releases should be created by merging from the `develop` branch into the `main` branch.

Before the merge is made the following should be done:

- The `CHANGELOG.md` file should be checked and updated with the version of the release
  and the date.
- The version should be updated in the following files:
  - `phantom/__init__.py`
  - `docs/conf.py`

After the merge is made the following should then be done:

- The release should be tagged with the version on GitHub.
- The announcement of the new release and it's changes should be distributed to
  interested users via relevant channels.
  - Major new features should be described in more detail than what is in the changelog.
  - For `MAJOR` releases, any backwards compatibility breaking changes must be
    prominently to users.
- TODO: packaging instructions.


### `MAJOR` Release Planning

Any backwards compatibility breaking changes in `MAJOR` releases should be announced/
discussed with users ahead of time. It may be preferable to create one or more beta
releases to allow users to test the changes and identify any potential serious issues
ahead of time.


## Building Documentation Pages

To build the documentation pages for Phantom run the following:

```sh
make doc
```

To view the pages in a web browser on your local machine run:

```sh
make host-doc
```

Then navigate to the URL displayed on the terminal in your web browser.


## Useful Links

- SemVer: Semantic Versioning Scheme:
    [https://semver.org]()

- PEP8: Official Python Style Guide:
    [https://www.python.org/dev/peps/pep-0008/]()

- Type annotations cheat sheet:
    [https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html]()

- 'Google Style' Python documentation guide:
    [https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html]()

- Black code formatter:
    [https://black.readthedocs.io/en/stable/]

- Sphinx Doc basics guide:
    [https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html]()