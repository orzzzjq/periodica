BAZEL ?= $(shell if command -v bazel >/dev/null 2>&1; then echo bazel; elif command -v bazelisk >/dev/null 2>&1; then echo bazelisk; else echo bazel; fi)
BAZEL_FLAGS ?= --enable_bzlmod
TARGET ?= //:periodica_so
SO_SRC ?= bazel-bin/_periodica.so
SO_DST ?= periodica/_periodica.so
VENV_DIR ?= .venv
VENV_PY ?= $(VENV_DIR)/bin/python
VENV_BIN ?= $(VENV_DIR)/bin
PYTHON_PACKAGES ?= numpy matplotlib scipy

.PHONY: all install-uv venv requirements build clean rebuild

all: build

install-uv:
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv already installed: $$(command -v uv)"; \
	else \
		echo "uv not found; installing via https://astral.sh/uv/install.sh"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -qO- https://astral.sh/uv/install.sh | sh; \
		else \
			echo "Error: curl or wget is required to install uv."; \
			exit 1; \
		fi; \
	fi

venv: install-uv
	@if [ -x "$(VENV_PY)" ]; then \
		echo "venv already exists at $(VENV_DIR); skipping creation"; \
	elif command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.11 $(VENV_DIR); \
	elif [ -x "$$HOME/.local/bin/uv" ]; then \
		$$HOME/.local/bin/uv venv --python 3.11 $(VENV_DIR); \
	else \
		echo "Error: uv is still not available after install."; \
		exit 1; \
	fi

requirements: venv
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(VENV_PY) $(PYTHON_PACKAGES); \
	elif [ -x "$$HOME/.local/bin/uv" ]; then \
		$$HOME/.local/bin/uv pip install --python $(VENV_PY) $(PYTHON_PACKAGES); \
	else \
		echo "Error: uv is still not available after install."; \
		exit 1; \
	fi

build:
	if [ -x "$(VENV_PY)" ]; then PATH="$(VENV_BIN):$$PATH" $(BAZEL) build $(BAZEL_FLAGS) $(TARGET); else $(BAZEL) build $(BAZEL_FLAGS) $(TARGET); fi
	cp -f $(SO_SRC) $(SO_DST)

clean:
	$(BAZEL) clean --expunge
	rm -f $(SO_DST)

rebuild: clean build
