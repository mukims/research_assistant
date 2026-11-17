PYTHON ?= python3
PORT ?= 8501

.PHONY: help install install-test test up down status docker-up docker-down dist clean

help:
	@echo "Marvin the Citebot — Developer & Distribution Targets"
	@echo ""
	@echo "  make install       Install full runtime dependencies + editable package"
	@echo "  make install-test  Install lightweight CI/test dependencies + editable package"
	@echo "  make test          Run the offline test suite"
	@echo "  make up            Start GROBID + Streamlit UI on 127.0.0.1 (via deploy/local.sh)"
	@echo "  make down          Stop the local GROBID container"
	@echo "  make status        Check status of app, LanceDB store, GROBID, and Ollama"
	@echo "  make docker-up     Build and start the containerized stack (Marvin + GROBID)"
	@echo "  make docker-down   Stop the containerized stack"
	@echo "  make dist          Build shareable source tarball and Python wheel in dist/"
	@echo "  make clean         Remove Python bytecode and test/build caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-test:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-test.txt
	$(PYTHON) -m pip install --no-deps -e .

test:
	CITATION_LOG_FILE=0 $(PYTHON) -m pytest tests/ -v

up:
	PYTHON="$(PYTHON)" PORT="$(PORT)" deploy/local.sh up

down:
	deploy/local.sh down

status:
	PORT="$(PORT)" deploy/local.sh status

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

dist: clean
	@mkdir -p dist
	git archive --format=tar.gz --prefix=research_assistant-0.1.0/ HEAD > dist/research_assistant-0.1.0-src.tar.gz
	$(PYTHON) -m pip wheel --no-deps --no-build-isolation -w dist .
	@ls -lh dist/

clean:
	find . -maxdepth 4 \( -name "__pycache__" -o -name ".pytest_cache" -o -name "*.egg-info" -o -name "build" \) -exec rm -rf {} +
	find . -maxdepth 4 -name "*.py[cod]" -delete
