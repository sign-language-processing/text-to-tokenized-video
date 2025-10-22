# SPDX-License-Identifier: Apache-2.0
# Post-training configuration for the DV8x16x16 tokenizer on PHOENIX-2014T data.
# Compatible with Cosmos Predict1 commit 2b446ebb36a7d19f22d293e0619be89ec2fbeb32 (Aug 2025).

from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.tokenizer.training.trainer import TokenizerTrainer
from cosmos_predict1.tokenizer.training.model import TokenizerModel
from cosmos_predict1.tokenizer.training.datasets.video_dataset import Dataset
from cosmos_predict1.utils.config import Config, JobConfig

def make_config():
    """Fine-tune the DV8x16x16 video tokenizer on PHOENIX sign language data."""

    # ---------------------------------------------------------
    # Model setup
    # ---------------------------------------------------------
    model = {
        "_target_": TokenizerModel,
        "config": {
            "network": {
                "_target_": "cosmos_predict1.tokenizer.networks.continuous_video.CausalContinuousVideoTokenizer",
                "z_channels": 3,  # input video channels (e.g. RGB)
                "z_factor": 2,  # scaling factor for latent channels
                "latent_channels": 8,  # latent size (adjust if model expects different)
                "encoder": "BASE",  # or a different variant if your checkpoint uses another
                "decoder": "BASE",
                "name": "CausalContinuousVideoTokenizer",
            },
            "loss": {
                "_target_": "cosmos_predict1.tokenizer.training.losses.continuous.ContinuousLoss",
            },
            "metric": {
                "_target_": "cosmos_predict1.tokenizer.training.metrics.recon.ReconstructionMetric",
            },
            "precision": "bfloat16",
            "ema": {
                "enabled": False,
                "beta": 0.9999,
                "torch_compile_buffer_renaming": False,
            },
        },
    }

    checkpoint = {
        "type": None,
        "load_path": "/data/mpanag/thesis_storage/checkpoints/Cosmos-Tokenize1-DV8x16x16-720p",
        "load_training_state": False,
        "only_load_scheduler_state": False,
        "strict_resume": True,
        "verbose": True,
        "async_saving": True,
        "jit": {  # ← this entire sub-block is required
            "enabled": False,
            "input_shape": None,
            "device": "cuda",
            "dtype": "bfloat16",
            "strict": True,
        },
    }

    # ---------------------------------------------------------
    # Dataset setup
    # ---------------------------------------------------------
    phoenix_root = "/scratch/mpanag/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"

    dataloader_train = L(Dataset)(
        dataset_path=phoenix_root,
        split="train",
        crop_size=(128, 128),
        clip_len=16,
        frame_stride=2,
        num_workers=8,
        batch_size=2,
        shuffle=True,
    )

    dataloader_val = L(Dataset)(
        dataset_path=phoenix_root,
        split="dev",
        crop_size=(128, 128),
        clip_len=16,
        frame_stride=2,
        num_workers=4,
        batch_size=2,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Trainer setup (plain dict version – safe for your case)
    # ---------------------------------------------------------
    trainer = {
        "type": TokenizerTrainer,  # required class reference
        "precision": "bf16",
        "epochs": 10,
        "log_every_n_steps": 20,
        "save_every": 1,
        "output_dir": "/data/mpanag/thesis_storage/checkpoints_posttrained/posttraining/tokenizer",
        "log_dir": "/data/mpanag/thesis_storage/logs/tokenizer_posttrain_phoenix",
        "seed": 42,
        "cudnn": {  # required by torch.backends.cudnn lines
            "deterministic": False,
            "benchmark": True,
        },
        "callbacks": {},  # required by CallBackGroup
        "timeout_period": 600,  # required by signal.signal(...timeout_period)
        "distributed_parallelism": "ddp",  # required by self.config.trainer.distributed_parallelism
        "grad_accum_iter": 1,  # required by training_step logic
        "memory_format": "contiguous_format",  # required by model.to(...memory_format)
        "grad_scaler_args": {},  # required by torch.amp.GradScaler(...)
        "ddp": {},  # required by distributed.parallel_model_wrapper(...)
        "run_validation": False,  # safe default to skip validation
        "validation_iter": 1000,  # needed if run_validation=True
        "max_iter": 10000,  # hard stop
    }

    # ---------------------------------------------------------
    # Job metadata (required by Config.validate)
    # ---------------------------------------------------------
    job = JobConfig(
        project="cosmos_tokenizer",
        group="phoenix_posttrain",
        name="dv8x16x16_128p_phoenix",
    )

    # Return a proper Config instance
    return Config(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        trainer=trainer,
        checkpoint=checkpoint,  # ← add this
        job=job,
    )

