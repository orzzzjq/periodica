BAZEL ?= $(shell if command -v bazel >/dev/null 2>&1; then echo bazel; elif command -v bazelisk >/dev/null 2>&1; then echo bazelisk; else echo bazelisk; fi)
BAZEL_FLAGS ?= --enable_bzlmod
TARGET ?= //:periodica_so
SO_SRC ?= bazel-bin/_periodica.so
SO_DST ?= periodica/_periodica.so
VENV_DIR ?= .venv
VENV_PY ?= $(VENV_DIR)/bin/python
VENV_BIN ?= $(VENV_DIR)/bin
PYTHON_PACKAGES ?= numpy matplotlib scipy fastapi 'uvicorn[standard]'
FRONTEND_DIR ?= web/frontend
FRONTEND_DIST ?= $(FRONTEND_DIR)/dist
WEB_HOST ?= 0.0.0.0
WEB_PORT ?= 8000

# ---------------------------------------------------------------------------
# Toolchain bootstrap (make setup)
#
# Tools that are not found on PATH are installed per-user, without root:
#   - with Homebrew when it is available and USE_BREW=1 (default on macOS),
#   - otherwise by downloading the official release into $(TOOLS_PREFIX)
#     (default ~/.local; binaries go to $(TOOLS_BIN) = ~/.local/bin).
# $(TOOLS_BIN) is prepended to PATH for every recipe in this Makefile, so a
# fresh install is picked up by later targets in the same `make` run.
# ---------------------------------------------------------------------------
TOOLS_PREFIX ?= $(HOME)/.local
TOOLS_BIN ?= $(TOOLS_PREFIX)/bin
NODE_PREFIX ?= $(TOOLS_PREFIX)/node
NODE_VERSION ?= 22.20.0
USE_BREW ?= $(if $(shell command -v brew 2>/dev/null),1,0)
export PATH := $(TOOLS_BIN):$(PATH)

UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  GO_ARCH := amd64
  NODE_ARCH := x64
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
  GO_ARCH := arm64
  NODE_ARCH := arm64
endif
BAZELISK_URL ?= https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-$(UNAME_S)-$(GO_ARCH)
NODE_URL ?= https://nodejs.org/dist/v$(NODE_VERSION)/node-v$(NODE_VERSION)-$(UNAME_S)-$(NODE_ARCH).tar.gz

# fetch <url> <dest-file>
define fetch
	if command -v curl >/dev/null 2>&1; then curl -fsSL "$(1)" -o "$(2)"; \
	elif command -v wget >/dev/null 2>&1; then wget -qO "$(2)" "$(1)"; \
	else echo "Error: curl or wget is required to download $(1)"; exit 1; fi
endef

.PHONY: all setup check-toolchain uv bazelisk node install-uv venv requirements build clean rebuild web web-deps web-build

all: setup build

# Install everything a fresh machine needs to build and run the project:
# uv (Python), bazelisk (C++ build), node/npm (web frontend), the Python venv
# with its packages, and the frontend's node_modules. Does not build (make all = setup + build).
setup: check-toolchain uv bazelisk node requirements web-deps
	@echo
	@echo "Setup complete:"
	@echo "  uv:       $$(command -v uv)"
	@echo "  bazelisk: $$(command -v bazelisk || command -v bazel)"
	@echo "  node:     $$(command -v node) ($$(node --version))"
	@echo "  npm:      $$(command -v npm) ($$(npm --version))"
	@echo "  venv:     $(VENV_DIR)"
	@case ":$$PATH:" in *":$(TOOLS_BIN):"*) ;; *) \
		echo; echo "NOTE: add $(TOOLS_BIN) to your PATH, e.g. for zsh/bash:"; \
		echo "  echo 'export PATH=\"$(TOOLS_BIN):\$$PATH\"' >> ~/.$${SHELL##*/}rc" ;; esac

# Bazel needs a C/C++ toolchain, which we cannot install without root; warn early.
check-toolchain:
	@if ! command -v cc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then \
		echo "WARNING: no C/C++ compiler found."; \
		if [ "$(UNAME_S)" = darwin ]; then echo "  Install the Xcode Command Line Tools: xcode-select --install"; \
		else echo "  Install one, e.g.: sudo apt-get install -y build-essential   (Debian/Ubuntu)"; fi; \
	fi
	@if ! command -v git >/dev/null 2>&1; then echo "WARNING: git not found; Bazel needs it to fetch some dependencies."; fi

uv:
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv already installed: $$(command -v uv)"; \
	else \
		echo "uv not found; installing via https://astral.sh/uv/install.sh"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$(TOOLS_BIN)" sh; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -qO- https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$(TOOLS_BIN)" sh; \
		else \
			echo "Error: curl or wget is required to install uv."; \
			exit 1; \
		fi; \
	fi

bazelisk:
	@if command -v bazelisk >/dev/null 2>&1 || command -v bazel >/dev/null 2>&1; then \
		echo "bazel already installed: $$(command -v bazelisk || command -v bazel)"; \
	elif [ "$(USE_BREW)" = 1 ]; then \
		echo "bazelisk not found; installing with Homebrew"; brew install bazelisk; \
	else \
		echo "bazelisk not found; downloading $(BAZELISK_URL)"; \
		mkdir -p "$(TOOLS_BIN)"; \
		$(call fetch,$(BAZELISK_URL),$(TOOLS_BIN)/bazelisk) && chmod +x "$(TOOLS_BIN)/bazelisk"; \
	fi

node:
	@if command -v npm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then \
		echo "node already installed: $$(command -v node) ($$(node --version))"; \
	elif [ "$(USE_BREW)" = 1 ]; then \
		echo "node/npm not found; installing with Homebrew"; brew install node; \
	else \
		echo "node/npm not found; downloading $(NODE_URL)"; \
		mkdir -p "$(TOOLS_BIN)" "$(NODE_PREFIX)"; \
		tmp="$$(mktemp)"; \
		$(call fetch,$(NODE_URL),$$tmp) && tar -xzf "$$tmp" -C "$(NODE_PREFIX)" --strip-components=1 && rm -f "$$tmp"; \
		for b in node npm npx; do ln -sf "$(NODE_PREFIX)/bin/$$b" "$(TOOLS_BIN)/$$b"; done; \
	fi

venv: uv
	@if [ -x "$(VENV_PY)" ]; then \
		echo "venv already exists at $(VENV_DIR); skipping creation"; \
	else \
		uv venv --python 3.11 $(VENV_DIR); \
	fi

requirements: venv
	uv pip install --python $(VENV_PY) $(PYTHON_PACKAGES)

build:
	if [ -x "$(VENV_PY)" ]; then PATH="$(VENV_BIN):$$PATH" $(BAZEL) build $(BAZEL_FLAGS) $(TARGET); else $(BAZEL) build $(BAZEL_FLAGS) $(TARGET); fi
	cp -f $(SO_SRC) $(SO_DST)

web-deps:
	@if ! command -v npm >/dev/null 2>&1; then \
		echo "Error: npm is required for the web frontend ($(FRONTEND_DIR)); run 'make setup'."; exit 1; \
	fi
	@if [ ! -d "$(FRONTEND_DIR)/node_modules" ] || [ "$(FRONTEND_DIR)/package.json" -nt "$(FRONTEND_DIR)/node_modules/.package-lock.json" ]; then \
		if [ -f "$(FRONTEND_DIR)/package-lock.json" ]; then npm --prefix $(FRONTEND_DIR) ci; \
		else npm --prefix $(FRONTEND_DIR) install; fi; \
	else \
		echo "frontend node_modules up to date"; \
	fi

web-build:
	@if command -v npm >/dev/null 2>&1; then \
		$(MAKE) web-deps; \
		npm --prefix $(FRONTEND_DIR) run build; \
	elif [ -d "$(FRONTEND_DIST)" ]; then \
		echo "npm not found; serving existing $(FRONTEND_DIST)"; \
	else \
		echo "Error: npm is required to build the web frontend ($(FRONTEND_DIR)); run 'make setup'."; \
		exit 1; \
	fi

web: web-build
	@if [ ! -x "$(VENV_BIN)/uvicorn" ] || [ ! -f "$(SO_DST)" ]; then \
		echo "Bootstrapping venv and native extension (first run)"; \
		$(MAKE) requirements build; \
	fi
	@echo "Web UI: http://localhost:$(WEB_PORT)"
	@if grep -qi microsoft /proc/version 2>/dev/null; then \
		( sleep 1; \
		  if command -v wslview >/dev/null 2>&1; then wslview "http://localhost:$(WEB_PORT)"; \
		  elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "http://localhost:$(WEB_PORT)" || true; \
		  fi ) >/dev/null 2>&1 & \
	fi
	$(VENV_BIN)/uvicorn app:app --app-dir web/server --host $(WEB_HOST) --port $(WEB_PORT) --reload

clean:
	$(BAZEL) clean --expunge
	rm -f $(SO_DST)

rebuild: clean build
