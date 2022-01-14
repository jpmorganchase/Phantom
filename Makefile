.DEFAULT_GOAL	:=help

MERCURY			:=mercury
PHANTOM			:=phantom

install-deps-mercury:
	cd $(MERCURY) && make install-deps && cd ..

install-deps-phantom:
	cd $(PHANTOM) && make install-deps && cd ..

install-deps: install-deps-mercury install-deps-phantom ## Install dependencies

install-mercury:
	cd $(MERCURY) && make install && cd ..

install-phantom:
	cd $(PHANTOM) && make install && cd ..

install: install-mercury install-phantom ## Install phantom

test-mercury:
	cd $(MERCURY) && make test && cd ..

test-phantom:
	cd $(PHANTOM) && make test && cd ..

test: test-mercury test-phantom ## Run the tests

format-mercury:
	cd $(MERCURY) && make format && cd ..

format-phantom:
	cd $(PHANTOM) && make format && cd ..

format: format-mercury format-phantom  ## Format the code

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "    \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: install test format help
