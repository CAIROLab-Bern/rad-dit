import wandb
import torch
from torchvision.utils import make_grid
from PIL import Image
import os
import argparse
import hashlib
import math

# Global flag to track if wandb is initialized
_wandb_initialized = False


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def init(project, entity=None, name=None, config=None, resume=False):
    """Initialize wandb (compatible with both dict config and argparse)."""
    global _wandb_initialized

    # Convert config to dict if it's an argparse Namespace
    if isinstance(config, argparse.Namespace):
        config_dict = namespace_to_dict(config)
    else:
        config_dict = config

    # Login if WANDB_KEY is provided, otherwise use existing login
    if "WANDB_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_KEY"])

    # Only set id and resume if explicitly resuming
    if resume:
        wandb.init(
            entity=entity,
            project=project,
            name=name,
            config=config_dict,
            id=generate_run_id(name),
            resume="allow",
        )
    else:
        # New run - don't set id, let wandb generate a new one
        wandb.init(
            entity=entity,
            project=project,
            name=name,
            config=config_dict,
        )
    _wandb_initialized = True


def initialize(args, entity, exp_name, project_name):
    """Legacy function for backward compatibility."""
    init(project=project_name, entity=entity, name=exp_name, config=args)


def log(stats, step=None):
    if _wandb_initialized:
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if _wandb_initialized:
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x