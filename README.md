# Text to Tokenized Video: Video-First Sign Language Generation

This repository contains the code and methodology for the Master's Thesis: **"Generating Sign Language Videos from Text using Fine-Tuned Video Token Representations."**

The goal is to develop an autoregressive language model that generates discrete video tokens directly from German text, utilizing a **domain-adapted NVIDIA Cosmos Tokenizer**.

---

## ⚙️ Project Setup and Development

### 1. Developer Tools (Local Workflow)

The repository uses standard Python tools for local development and quality checks:

| Command | Description |
| :--- | :--- |
| `pip install ".[dev]"` | Install development dependencies, including testing tools. |
| `ruff check . --fix` | Run linting and formatting checks. |
| `pytest .` | Execute unit tests to verify pipeline integrity. |

### 2. Required External Resources (Data Hygiene)

| Asset | Acquisition Method | Expected Local Path |
| :--- | :--- | :--- |
| **RWTH-PHOENIX-2014-T Dataset** | Follow instructions on the official project page. | `/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/` |
| **Cosmos DV8x16x16 Checkpoints** | Download the pre-trained weights from Hugging Face. | `checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/` |

**Checkpoint Download Instruction:**
```bash
huggingface-cli download nvidia/Cosmos-1.0-Tokenizer-DV8x16x16 --local-dir checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16
```

### 3. Environment and Dependencies

1.  **Clone the NVIDIA Cosmos-Predict1 framework** (used as a library/CLI) into the project root:
    ```bash
    git clone [https://github.com/nvidia-cosmos/cosmos-predict1.git](https://github.com/nvidia-cosmos/cosmos-predict1.git)
    ```
2.  **Activate your Conda Environment** (the one containing your PyTorch 2.6.0 stack):
    ```bash
    conda activate cosmos
    ```
3.  **Install project dependencies**:
    ```bash
    pip install -r requirements.txt
    # CRITICAL: Set PYTHONPATH to resolve local module imports during torchrun.
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

---

## Usage 

### A. Tokenize the Dataset (Data Preparation)

This uses the custom script (`encode_dataset.py`) to handle PHOENIX-specific preprocessing (resizing, chunking, padding) and saves the discrete **$\mathbf{FSQ}$ video tokens**.

### B. Decode Reconstructed Video (Evaluation)

Use the project script (`decode_tokens.py`) to decode the tokens back into video for visual and quantitative quality checks.

```bash
# Example: Decode a token file using the decoder checkpoint.
python decode_tokens.py \
    --token_path /path/to/token_output/sequence.pt \
    --checkpoint_dec checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit \
    --output_path /path/to/reconstructed_video.mp4
```