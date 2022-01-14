.DEFAULT_GOAL	:=help

MODULE			:=phantom
MYPY_CONF		:=mypy.ini

PYTHON			:=python3
PYTEST			:=$(PYTHON) -m pytest
PIP				:=$(PYTHON) -m pip
BLACK			:=$(PYTHON) -m black
MYPY			:=$(PYTHON) -m mypy

install_deps:  ## Install dependencies
	$(PIP) install -r requirements.txt

install_dev_deps:  ## Install dev dependencies
	$(PIP) install -r requirements-dev.txt

install: ## Install the package
	$(PYTHON) setup.py install

test:  ## Run the tests
	$(PYTEST) tests

cov:  ## Run the tests with coverage
	$(PYTEST) tests --cov-report term-missing:skip-covered --cov=${MODULE}

check:  ## Check the types
	$(MYPY) --config-file ${MYPY_CONF} ${MODULE}

format:  ## Format the code
	$(BLACK) ${MODULE} tests

dev:  ## Build the package in develop mode
	$(PYTHON) setup.py develop --user -O2

clean:  ## Clean the worspace
	rm -rf $(MODULE).egg-info/ .pytest_cache/ .mypy_cache/

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "    \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: check test clean help
