import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.distributed as dist

from model import ModelArgs, Transformer
from safetensors.torch import load_model


def setup_logging(rank, log_dir=None):
    """
    Set up logging configuration for distributed runs.

    Args:
        rank (int): Process rank
        log_dir (str, optional): Directory to save log files. If None, logs to console only.
    """
    # Create formatter
    formatter = logging.Formatter(
        f"[%(asctime)s][rank{rank}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Always add console handler for rank 0
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}_rank{rank}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


@torch.inference_mode()
def run_single_forward(
    model: Transformer,
    seq_len: int = 2048,
    batch_size: int = 1,
    vocab_size: int = 120000,  # Default DeepSeek-V3 vocab size is 129280
) -> None:
    """
    Runs a single forward pass with random initialized tokenized inputs.

    Args:
        model (Transformer): The transformer model to use.
        seq_len (int, optional): Length of the input sequence. Defaults to 2048.
        batch_size (int, optional): Batch size for the forward pass. Defaults to 1.
        vocab_size (int, optional): Vocabulary size for random token generation. Defaults to 128000.
    """
    logging.info(
        f"Running single forward pass with sequence length {seq_len}, batch size {batch_size}"
    )

    # Create random token inputs (values between 1 and vocab_size-1)
    torch.manual_seed(42)
    tokens = torch.randint(
        1, vocab_size, (batch_size, seq_len), dtype=torch.long, device="cuda"
    )

    logging.info(f"Fake input token IDs: {tokens[0][:10].cpu().numpy()}...")
    logging.info(f"Fake input shape: {tokens.shape}")
    logging.info(f"Input tensors device: {tokens.device}")

    # Measure time for the forward pass
    start_time = time.time()

    # Run forward pass
    logits = model.forward(tokens, 0)

    # Force synchronization to get accurate timing
    torch.cuda.synchronize()
    end_time = time.time()

    # Log statistics about the output
    logging.info(f"Forward pass completed in {end_time - start_time:.4f} seconds")
    logging.info(f"Output shape: {logits.shape}")
    logging.info(
        f"Output stats - Mean: {logits.mean().item():.4f}, "
        f"Min: {logits.min().item():.4f}, "
        f"Max: {logits.max().item():.4f}"
    )

    # Calculate and log memory usage
    memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
    memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
    logging.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    logging.info(f"GPU memory reserved: {memory_reserved:.2f} GB")

    return logits


def main(
    ckpt_path: str,
    config: str,
    seq_len: int = 2048,
    batch_size: int = 1,
    log_dir: str = None,
) -> None:
    """
    Main function to load the model and perform a single forward pass.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        seq_len (int, optional): Length of the input sequence. Defaults to 2048.
        batch_size (int, optional): Batch size for the forward pass. Defaults to 1.
        log_dir (str, optional): Directory to save log files. If None, logs to console only.
    """
    # Initialize distributed environment if needed
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Set up logging
    logger = setup_logging(rank, log_dir)

    if world_size > 1:
        dist.init_process_group("nccl")

    logging.info(
        f"Starting process with rank {rank}/{world_size} (local_rank: {local_rank})"
    )

    # Set up CUDA and random seed
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(42)

    # Log GPU info
    device_name = torch.cuda.get_device_name(local_rank)
    logging.info(f"Using device: {device_name} (CUDA:{local_rank})")

    # Load model configuration
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    logging.info(f"Model configuration: {args}")

    # Initialize model
    with torch.device("cuda"):
        model = Transformer(args)

    # Load model weights if checkpoint path is provided
    if ckpt_path:
        logging.info(f"Loading model from {ckpt_path}")
        # Check if the checkpoint path is a directory or a file
        if os.path.isdir(ckpt_path):
            # Try to find the correct safetensors file
            possible_patterns = [
                f"model{rank}-mp{world_size}.safetensors",  # Standard pattern
                f"model.safetensors",  # Single file pattern
                f"consolidated.{rank:02d}.safetensors",  # HF PEFT pattern
            ]

            model_file = None
            for pattern in possible_patterns:
                file_path = os.path.join(ckpt_path, pattern)
                if os.path.exists(file_path):
                    model_file = file_path
                    break

            if model_file is None:
                # List available files to help diagnose
                logging.warning(f"Available files in {ckpt_path}:")
                for file in os.listdir(ckpt_path):
                    if file.endswith(".safetensors"):
                        logging.warning(f"  - {file}")
                raise FileNotFoundError(
                    f"Could not find a suitable model file in {ckpt_path}"
                )

            logging.info(f"Loading from file: {model_file}")
            # Correct way to call load_model with the model as first argument
            load_model(model, model_file)
        else:
            # Direct file path provided
            logging.info(f"Loading from file: {ckpt_path}")
            load_model(model, ckpt_path)
    else:
        logging.info("Using randomly initialized model weights")

    # Run the forward pass
    run_single_forward(model, seq_len, batch_size)

    logging.info(f"Rank {rank} completed successfully")

    # Clean up
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt-path", type=str, default="/data/users/jianiw/dsv3-weights-5-layer/"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files. If not provided, logs will be saved to './logs'",
    )
    args = parser.parse_args()

    main(args.ckpt_path, args.config, args.seq_len, args.batch_size, args.log_dir)

# torchrun --nnodes 1 --nproc-per-node 8 run_single_forward.py --config configs/config_671B.json
