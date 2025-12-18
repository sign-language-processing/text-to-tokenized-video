.PHONY: install install-dev download-checkpoints lint lint-fix format test clean \
        encode decode tokenize-all \
        build-datasets build-text2video build-video2gloss build-gloss2video build-gloss2text \
        train-text2video train-video2gloss train-gloss2video train-gloss2text help

# Default checkpoint paths
CHECKPOINT_DIR ?= checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16
ENCODER_CKPT ?= $(CHECKPOINT_DIR)/encoder.jit
DECODER_CKPT ?= $(CHECKPOINT_DIR)/decoder.jit

# =============================================================================
# Installation
# =============================================================================

install:  ## Install package with Cosmos tokenizer
	pip install git+https://github.com/nvidia-cosmos/cosmos-predict1.git --no-deps
	pip install ".[dev]"

install-dev:  ## Install development dependencies only
	pip install ".[dev]"

download-checkpoints:  ## Download Cosmos tokenizer checkpoints from HuggingFace
	huggingface-cli login
	huggingface-cli download nvidia/Cosmos-1.0-Tokenizer-DV8x16x16 \
		--local-dir $(CHECKPOINT_DIR)

# =============================================================================
# Development
# =============================================================================

lint:  ## Run linting with ruff
	ruff check .

lint-fix:  ## Run linting and auto-fix issues
	ruff check . --fix

format:  ## Format code with ruff
	ruff format .

test:  ## Run tests with pytest
	pytest tests/ -v

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Video Tokenization
# =============================================================================

encode:  ## Encode video to tokens (VIDEO=input.mp4 OUTPUT=tokens.pt)
ifndef VIDEO
	$(error VIDEO is required. Usage: make encode VIDEO=input.mp4 OUTPUT=tokens.pt)
endif
ifndef OUTPUT
	$(error OUTPUT is required. Usage: make encode VIDEO=input.mp4 OUTPUT=tokens.pt)
endif
	python cosmos_tokenizer/cosmos_tokenizer.py encode \
		--video $(VIDEO) \
		--checkpoint-enc $(ENCODER_CKPT) \
		--output $(OUTPUT)

decode:  ## Decode tokens to video (TOKENS=tokens.pt OUTPUT=video.mp4 FPS=25)
ifndef TOKENS
	$(error TOKENS is required. Usage: make decode TOKENS=tokens.pt OUTPUT=video.mp4)
endif
ifndef OUTPUT
	$(error OUTPUT is required. Usage: make decode TOKENS=tokens.pt OUTPUT=video.mp4)
endif
	python cosmos_tokenizer/cosmos_tokenizer.py decode \
		--tokens $(TOKENS) \
		--checkpoint-dec $(DECODER_CKPT) \
		--output $(OUTPUT) \
		--fps $(or $(FPS),25)

tokenize-all:  ## Tokenize entire PHOENIX dataset
	./cosmos_tokenizer/tokenize_all.sh

# =============================================================================
# Dataset Building
# =============================================================================

build-text2video:  ## Build text-to-video JSONL dataset
	python scripts/text2video/text2video_datasetbuilder.py

build-video2gloss:  ## Build video-to-gloss JSONL dataset
	python scripts/video2gloss/video2gloss_datasetbuilder.py

build-gloss2video:  ## Build gloss-to-video JSONL dataset
	python scripts/gloss2video/gloss2video_datasetbuilder.py

build-gloss2text:  ## Build gloss-to-text JSONL dataset
	python scripts/gloss2text/gloss2text_datasetbuilder.py

build-datasets: build-text2video build-video2gloss build-gloss2video build-gloss2text  ## Build all datasets

# =============================================================================
# T5 Model Training
# =============================================================================

train-text2video:  ## Train text-to-video T5 model (VARIANT=t5-small-36|t5-small-100|t5-base)
	cd scripts && ./text2video/script.sh $(or $(VARIANT),t5-small-36)

train-video2gloss:  ## Train video-to-gloss T5 model (VARIANT=t5-small|t5-base)
	cd scripts && ./video2gloss/script.sh $(or $(VARIANT),t5-small)

train-gloss2video:  ## Train gloss-to-video T5 model
	cd scripts && ./gloss2video/script.sh

train-gloss2text:  ## Train gloss-to-text T5 model
	cd scripts && ./gloss2text/script.sh

# =============================================================================
# Help
# =============================================================================

help:  ## Show this help message
	@echo "Text-to-Tokenized-Video - Sign Language Video Generation"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
