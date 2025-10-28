install:
	pip install git+https://github.com/nvidia-cosmos/cosmos-predict1.git --no-deps
	pip install ".[dev]"

download_checkpoints:
	hf auth login
	hf download nvidia/Cosmos-Tokenize1-DV8x16x16-720p --local-dir checkpoints/Cosmos-Tokenize1-DV8x16x16-720p
