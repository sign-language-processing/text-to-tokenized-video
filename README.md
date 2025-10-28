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

### Download the Phoenix Dataset

```shell
# Download the PHOENIX-2014T v3 release
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz

# Extract the dataset archive
tar -xvzf phoenix-2014-T.v3.tar.gz
```

### Tokenizer Example

Encode a video:

```shell
python -m text_to_tokenized_video.tokenizer.encode_video \
    --video=assets/example.mp4 \
    --checkpoint-enc=checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/encoder.jit \
    --output-path=test.pt
 ```

Encode a dataset:

```shell
export PHOENIX="PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
export TOKENS_DIR="tokens"
./text_to_tokenized_video/tokenizer/tokenize_dataset.sh
```