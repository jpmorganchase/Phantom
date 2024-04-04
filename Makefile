.DEFAULT_GOAL	:=help

MODULE			:=phantom

PYTHON			:=python3
PYTEST			:=$(PYTHON) -m pytest
PIP				:=$(PYTHON) -m pip
POETRY			:=$(PYTHON) -m poetry
SPHINXBUILD     :=$(PYTHON) -m sphinx


install: ## Install the package
	$(PIP) install .

install-dev:  ## Install dev dependencies
	$(PIP) install poetry && $(POETRY) install --with dev

test:  ## Run the tests
	$(PYTEST) tests

cov:  ## Run the tests with coverage
	$(PYTEST) tests --cov-report term-missing:skip-covered --cov=${MODULE}

dev:  ## Build the package in develop mode
	$(PIP) install --editable .

build-docs:  ## Build the documentation
	@$(SPHINXBUILD) -M html "docs/" "docs/_build"

host-docs:  ## Host the documentation locally
	@$(PYTHON) -m http.server --directory docs/_build/html

clean:  ## Clean the worspace
	rm -rf $(MODULE).egg-info/ .pytest_cache/ .mypy_cache/

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "    \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: check test clean help
