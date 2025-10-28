# Text to Tokenized Video

## Usage

Installation:
```shell
# Install dependencies
make install
# Download Necessary Checkpoints
make download_checkpoints
```

Lint:
```shell
ruff check . --fix
```

Test:
```shell
pytest .
```

### Tokenizer Example

Encode a video:
```shell
 python -m text_to_tokenized_video.tokenizer.encode_video \
    --video=assets/example.mp4 \
    --checkpoint-enc=checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/encoder.jit \
    --output-path=test.txt
 ```