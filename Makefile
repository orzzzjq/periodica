BAZEL ?= $(shell if command -v bazel >/dev/null 2>&1; then echo bazel; elif command -v bazelisk >/dev/null 2>&1; then echo bazelisk; else echo bazel; fi)
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

.PHONY: all install-uv venv requirements build clean rebuild web web-build

all: requirements build

uv:
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

venv: uv
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

web-build:
	@if command -v npm >/dev/null 2>&1; then \
		if [ ! -d "$(FRONTEND_DIR)/node_modules" ] || [ "$(FRONTEND_DIR)/package.json" -nt "$(FRONTEND_DIR)/node_modules/.package-lock.json" ]; then \
			npm --prefix $(FRONTEND_DIR) install; \
		fi; \
		npm --prefix $(FRONTEND_DIR) run build; \
	elif [ -d "$(FRONTEND_DIST)" ]; then \
		echo "npm not found; serving existing $(FRONTEND_DIST)"; \
	else \
		echo "Error: npm is required to build the web frontend ($(FRONTEND_DIR))."; \
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
