# Text to Tokenized Video

**Generating Sign Language Videos from Text Using Tokenised Video Representations**

Bachelor's Thesis - University of Zurich, Department of Computational Linguistics

This repository explores the feasibility of generating sign language video directly from spoken language input using discrete video token representations. Unlike traditional pose-based pipelines, this approach bypasses skeletal intermediates and operates directly on video tokens produced by NVIDIA's Cosmos tokenizer.

## Overview

The pipeline translates German text into German Sign Language (DGS) videos through discrete video tokens:

```
German Text --> T5 (Seq2Seq) --> Discrete Video Tokens --> Cosmos Decoder --> Sign Language Video
```

### Research Questions

1. Can a general-purpose video tokenizer (Cosmos) effectively represent sign language content?
2. Can a small language model learn to generate token sequences that reconstruct coherent signing?
3. What are the limitations of token-based video generation for linguistically rich domains like sign language?

### Key Findings

- The Cosmos tokenizer retains coarse gestural structure and motion dynamics, though fine-grained handshapes and facial expressions are degraded
- T5 models achieved BLEU scores of 7-11 on text-to-video token generation, with T5-base outperforming T5-small
- Reconstructed videos from predicted tokens remain visually incoherent, highlighting the challenge of preserving spatial structure in autoregressive generation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/text-to-tokenized-video.git
cd text-to-tokenized-video

# Install dependencies
make install

# Download Cosmos tokenizer checkpoints (requires HuggingFace login)
make download-checkpoints
```

### Manual Installation

```bash
# Install Cosmos tokenizer
pip install git+https://github.com/nvidia-cosmos/cosmos-predict1.git --no-deps
pip install ".[dev]"

# Download checkpoints
huggingface-cli login
huggingface-cli download nvidia/Cosmos-1.0-Tokenizer-DV8x16x16 \
    --local-dir checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16
```

## Project Structure

```
text-to-tokenized-video/
├── cosmos_tokenizer/                 # Cosmos video tokenization tools
│   ├── cosmos_tokenizer.py           # Unified encode/decode interface
│   └── tokenize_all.sh               # Batch tokenize full dataset
├── scripts/                          # Data processing & training
│   ├── run_translation.py            # HuggingFace T5 training script
│   ├── text2video/                   # German text --> video tokens
│   │   ├── text2video_datasetbuilder.py
│   │   ├── script.sh                 # Train T5-small/T5-base
│   │   └── reshape_example_video.py
│   ├── video2gloss/                  # Video tokens --> gloss
│   │   ├── video2gloss_datasetbuilder.py
│   │   ├── script.sh                 # Train T5 for video-to-gloss
│   │   └── lstm/                     # LSTM baseline experiments
│   │       ├── video2gloss_train_lstm.py
│   │       └── video2gloss_eval_lstm.py
│   ├── gloss2video/                  # Gloss --> video tokens
│   │   ├── gloss2video_datasetbuilder.py
│   │   └── script.sh
│   └── gloss2text/                   # Gloss --> German text
│       ├── gloss2text_datasetbuilder.py
│       └── script.sh
├── pyproject.toml
└── Makefile
```

## Cosmos Tokenizer

This project uses **NVIDIA Cosmos-1.0-Tokenizer-DV8x16x16**, a discrete video tokenizer that compresses video into a compact token representation.

### How It Works

The Cosmos tokenizer uses a **Finite Scalar Quantization (FSQ)** approach:

1. **Encoder**: Compresses video frames into a latent space
2. **Quantizer**: Maps continuous latents to discrete token indices
3. **Decoder**: Reconstructs video from discrete tokens

**Compression ratio**: A 128x128 video at 25 FPS is compressed to:
- Temporal: 8x compression (8 frames → 1 token row)
- Spatial: 16x16 compression (128x128 → 8x8 tokens per frame)

**Token shape**: `[1, T, 8, 8]` where T = num_frames // 8

### CLI Usage

#### Encode a Video to Tokens

```bash
python cosmos_tokenizer/cosmos_tokenizer.py encode \
    --video input.mp4 \
    --checkpoint-enc checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit \
    --output tokens.pt \
    --height 128 \
    --width 128
```

The encoder accepts either:
- A video file (`.mp4`, `.avi`, etc.)
- A directory of PNG frames

**Output**: A `.pt` file containing:
- `indices`: Token indices with shape `[1, T, 8, 8]`
- `codes`: FSQ codes with shape `[1, 6, T, 8, 8]`

#### Decode Tokens to Video

```bash
python cosmos_tokenizer/cosmos_tokenizer.py decode \
    --tokens tokens.pt \
    --checkpoint-dec checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit \
    --output reconstructed.mp4 \
    --fps 25
```

#### Batch Processing

Tokenize the entire PHOENIX dataset:

```bash
# Set environment variables (optional, has defaults)
export PHOENIX_DIR=~/scratch/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px
export TOKENS_DIR=~/data/thesis_storage/tokens

# Run batch tokenization
./cosmos_tokenizer/tokenize_all.sh
```


### Makefile Shortcuts

```bash
# Encode a single video
make encode VIDEO=input.mp4 OUTPUT=tokens.pt

# Decode tokens to video
make decode TOKENS=tokens.pt OUTPUT=video.mp4 FPS=25
```

## Usage

### Step 1: Tokenize the PHOENIX Dataset

Encode sign language videos into discrete tokens:

```bash
# Using the batch script
./cosmos_tokenizer/tokenize_all.sh

# Or using Python directly
python scripts/tokenize_dataset.py --limit 100  # Test with 100 samples
python scripts/tokenize_dataset.py              # Full dataset
```

### Step 2: Build Training Datasets

Each experiment direction has its own dataset builder:

```bash
# Text --> Video tokens (main task)
python scripts/text2video/text2video_datasetbuilder.py

# Video tokens --> Gloss
python scripts/video2gloss/video2gloss_datasetbuilder.py

# Gloss --> Video tokens
python scripts/gloss2video/gloss2video_datasetbuilder.py

# Gloss --> Text
python scripts/gloss2text/gloss2text_datasetbuilder.py
```

Output format (JSONL):
```json
{"translation": {"en": "<source>", "de": "<target>"}}
```

### Step 3: Train the T5 Model

Fine-tune T5 for sequence-to-sequence translation:

```bash
# Text-to-Video with T5-small (36 epochs)
cd scripts && ./text2video/script.sh t5-small-36

# Text-to-Video with T5-small (100 epochs)
cd scripts && ./text2video/script.sh t5-small-100

# Text-to-Video with T5-base (36 epochs)
cd scripts && ./text2video/script.sh t5-base
```

Or use the training script directly:

```bash
python scripts/run_translation.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang de \
    --train_file text_to_video_train.jsonl \
    --validation_file text_to_video_dev.jsonl \
    --test_file text_to_video_test.jsonl \
    --output_dir output_text2video \
    --predict_with_generate \
    --max_source_length 512 \
    --max_target_length 2048 \
    --num_train_epochs 36 \
    --report_to wandb
```

### Step 4: Decode Predicted Tokens

Reconstruct videos from T5-predicted tokens:

```bash
python cosmos_tokenizer/cosmos_tokenizer.py decode \
    --tokens predicted_tokens.pt \
    --checkpoint-dec checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit \
    --output reconstructed.mp4 \
    --fps 25
```

## Experiments

Five translation directions were explored:

| Experiment | Mapping | Purpose |
|------------|---------|---------|
| **Text-to-Video** | German text → Video tokens | Core task: direct text-to-video generation |
| Video-to-Gloss | Video tokens → Gloss | Assess semantic preservation in tokenization |
| Gloss-to-Video | Gloss → Video tokens | Test if explicit supervision simplifies generation |
| Text-to-Gloss | German text → Gloss | Baseline for linguistic modeling |
| Gloss-to-Text | Gloss → German text | Sanity check for dataset alignment |

### Results

| Model | Task | Epochs | Val BLEU | Test BLEU |
|-------|------|--------|----------|-----------|
| T5-small | Text→Video | 36 | 5.75 | 4.68 |
| T5-small | Text→Video | 100 | 6.51 | 7.31 |
| T5-base | Text→Video | 36 | 9.97 | 10.87 |

## Dataset

This work uses the **RWTH-PHOENIX-Weather 2014T** corpus:
- 8,257 training / 825 dev / 1,004 test sentences
- German Sign Language (DGS) weather forecasts
- Parallel video, gloss, and German text annotations
- 210x260 pixels at 25 FPS (resized to 128x128 for tokenization)

## Limitations

- **Tokenization compression**: 128x128 input to 8x8 tokens per frame loses fine-grained detail
- **Domain mismatch**: Cosmos was not trained on sign language data
- **Sequence flattening**: 4D token grids flattened to 1D lose spatial structure during autoregressive generation
- **Evaluation metrics**: BLEU is not well-suited for discrete video token sequences

## Development

```bash
# Install dev dependencies
pip install ".[dev]"

# Run linting
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format
```

## External Resources

| Resource | Description |
|----------|-------------|
| [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | Sign language dataset |
| [Cosmos Tokenizer](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16) | Video tokenization |
| [T5 Models](https://huggingface.co/google-t5) | Sequence-to-sequence model |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Training framework |

## Citation

```bibtex
@bachelorsthesis{panagiotopoulou2025signlang,
    title={Generating Sign Language Videos from Text Using Tokenised Video Representations},
    author={Panagiotopoulou, Maria Christina},
    year={2025},
    school={University of Zurich},
    department={Department of Computational Linguistics}
}
```

## Acknowledgments

- **Supervisor**: Prof. Dr. Sarah Ebling
- **Technical Advisor**: Dr. Amit Moryossef
- NVIDIA for the Cosmos Tokenizer
- RWTH Aachen for the PHOENIX-2014-T dataset
- HuggingFace for Transformers and T5 models

## License

This project is for academic research purposes. See individual component licenses:
- Cosmos Tokenizer: [NVIDIA License](https://github.com/NVIDIA/Cosmos-Tokenizer)
- PHOENIX Dataset: Research use only
- HuggingFace Transformers: Apache 2.0
