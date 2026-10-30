"""
Training Codes of LightningDiT with Sharded Latent Dataset.

Integrates a shard loader for memory-efficient training on large-scale datasets.
Adds WandB logging for better experiment tracking.

Based on original train.py by Maple (Jingfeng Yao) from HUST-VL
Adapted for sharded data loading.
"""

import os
import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from tokenizer.vavae import VA_VAE
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import shardloader components
from shardloader import (
    ShardedLatentDataset, ShardWindowSampler,
    collate_has_tumor_extended, collate_region_extended,
)

# Import wandb utilities
import wandb_utils


def load_vae_for_sampling(vae_name, device, logger):
    """Load VA-VAE for sampling visualization."""
    vae = VA_VAE(f'tokenizer/configs/{vae_name}.yaml').model
    vae = vae.to(device)
    vae.eval()

    logger.info(f"Loaded VAE '{vae_name}' for sampling")
    return vae


def decode_with_vae(vae, latents, vae_name, latent_mean=None, latent_std=None, latent_multiplier=1.0, batch_size=1):
    """
    Decode latents using appropriate scaling/normalization for the VAE type.
    Decodes in smaller batches to avoid OOM.
    Immediately moves decoded batches to CPU to free GPU memory.

    CRITICAL: The model generates latents in the SAME space as training data.
    - If training data was normalized (VA-VAE), generated latents are normalized too.
    - Must DENORMALIZE before VAE decoding!

    Training normalization (img_latent_dataset.py line 83-85):
        z_train = ((z_raw - mean) / std) * latent_multiplier

    Decoding denormalization (inference.py line 181, 220):
        z_decode = (z_generated * std) / latent_multiplier + mean

    Args:
        vae: VAE decoder model
        latents: Generated latent tensors [B, C, H, W] (in normalized space if VA-VAE)
        vae_name: Name of VAE model
        latent_mean: Channel-wise mean [C, 1, 1] (for VA-VAE denormalization)
        latent_std: Channel-wise std [C, 1, 1] (for VA-VAE denormalization)
        latent_multiplier: Multiplier used during training (default: 1.0)
        batch_size: Decode in batches of this size to save memory (default: 4)

    Returns:
        Decoded images [B, 3, H, W] in range [-1, 1] ON CPU
    """
    # Denormalize latents based on VAE type
    if 'vavae' in vae_name.lower():
        # VA-VAE: Latents are in NORMALIZED space, must denormalize!
        # Formula matches inference.py line 181, 220:
        # z_decode = (z_normalized * std) / latent_multiplier + mean
        if latent_mean is not None and latent_std is not None:
            latents = (latents * latent_std.to(latents.device)) / latent_multiplier + latent_mean.to(latents.device)
        else:
            # ERROR: VA-VAE requires stats for denormalization
            raise ValueError("VA-VAE requires latent_mean and latent_std for denormalization!")

    # Decode in batches to avoid OOM
    # CRITICAL: Move each batch to CPU immediately to free GPU memory
    decoded_images = []
    num_batches = (latents.shape[0] + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, latents.shape[0], batch_size)):
        batch_latents = latents[i:i+batch_size]
        print(f"    [DEBUG] Decoding batch {batch_idx+1}/{num_batches}, shape={batch_latents.shape}")

        # Check memory before decode
        allocated = torch.cuda.memory_allocated(batch_latents.device) / 1024**3
        print(f"    [DEBUG] Before decode: {allocated:.2f}GB allocated")

        batch_decoded = vae.decode(batch_latents)

        # Move to CPU immediately and free GPU memory
        decoded_images.append(batch_decoded.detach().cpu())
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated(batch_latents.device) / 1024**3
        print(f"    [DEBUG] After decode + CPU transfer: {allocated:.2f}GB allocated")

    # Concatenate on CPU
    print(f"    [DEBUG] Concatenating {len(decoded_images)} batches on CPU")
    return torch.cat(decoded_images, dim=0)


def generate_samples_for_visualization(
    ema_model, transport, vae, vae_name,
    num_samples, latent_size, latent_channels, image_size,
    num_classes, cfg_scale, device, logger,
    latent_mean=None, latent_std=None, latent_multiplier=1.0,
    num_steps=50,
):
    """
    Generate samples for training visualization.
    Returns tensor of images [N, 3, H, W] in range [-1, 1].

    For extended ImageNet models (num_classes > 1000), generates:
    - Mostly medical class samples (1001: no_tumor, 1002: has_tumor)
    - A few random ImageNet samples for comparison (5 samples)

    Args:
        latent_mean: Channel-wise mean [C, 1, 1] for denormalization (VA-VAE)
        latent_std: Channel-wise std [C, 1, 1] for denormalization (VA-VAE)
        latent_multiplier: Multiplier used during training (default: 1.0)
        num_steps: Number of ODE sampling steps (default: 50)
    """
    transport_sampler = Sampler(transport)
    use_cfg = cfg_scale > 1.0
    cfg_token_idx = num_classes  # CFG token at end

    # Prepare labels
    if num_classes > 1000:
        # Extended-on-ImageNet: medical classes are the appended block
        # {1001 .. num_classes-1}. has_tumor_extended → {1001,1002};
        # region_extended → {1001..1005}. Cycle through ALL of them so the
        # viz grid is representative for every appended class.
        num_imagenet_samples = 3
        num_medical_samples = num_samples - num_imagenet_samples

        medical_classes = torch.arange(1001, num_classes, device=device)
        n_med_cls = max(int(medical_classes.numel()), 1)
        medical_ys = medical_classes.repeat(
            (num_medical_samples + n_med_cls - 1) // n_med_cls
        )[:num_medical_samples]

        # 5 random ImageNet samples (one per random class)
        imagenet_ys = torch.randint(0, 1000, size=(num_imagenet_samples,), device=device)

        # Combine: medical first, then ImageNet
        ys = torch.cat([medical_ys, imagenet_ys], dim=0)

        logger.info(f"  Sampling: {num_medical_samples} medical + {num_imagenet_samples} ImageNet classes")
    else:
        # Standalone model: sample from all classes
        ys = torch.randint(num_classes, size=(num_samples,), device=device)

    # Prepare latent noise
    logger.info(f"  [DEBUG] Creating latent noise: {num_samples} samples, {latent_channels}ch, {latent_size}x{latent_size}")
    zs = torch.randn(num_samples, latent_channels, latent_size, latent_size, device=device)

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"  [DEBUG] After creating zs: {allocated:.2f}GB allocated")

    if use_cfg:
        logger.info(f"  [DEBUG] Enabling CFG with scale={cfg_scale}, doubling batch to {num_samples*2}")
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([cfg_token_idx] * num_samples, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=cfg_scale)
        model_fn = lambda *args, **kwargs: ema_model.forward_with_cfg(*args, **kwargs)

        allocated = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"  [DEBUG] After CFG doubling: {allocated:.2f}GB allocated, zs.shape={zs.shape}")
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = lambda *args, **kwargs: ema_model(*args, **kwargs)

    # Generate latent samples
    logger.info(f"  [DEBUG] Starting ODE sampling with {num_steps} steps...")
    sample_fn = transport_sampler.sample_ode(num_steps=num_steps)
    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"  [DEBUG] After ODE sampling: {allocated:.2f}GB allocated, samples.shape={samples.shape}")

    if use_cfg:
        samples, _ = samples.chunk(2, dim=0)
        logger.info(f"  [DEBUG] After CFG chunk: samples.shape={samples.shape}")

    # Free GPU memory from sampling (zs, ys no longer needed)
    logger.info(f"  [DEBUG] Freeing zs, ys...")
    del zs, ys
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"  [DEBUG] After freeing zs/ys: {allocated:.2f}GB allocated")

    # Decode to images (with denormalization for VA-VAE)
    # decode_with_vae returns CPU tensors and frees GPU memory batch-by-batch
    logger.info(f"  [DEBUG] Starting VAE decode (samples.shape={samples.shape})...")
    with torch.no_grad():
        images = decode_with_vae(vae, samples, vae_name, latent_mean, latent_std, latent_multiplier)

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"  [DEBUG] After VAE decode: {allocated:.2f}GB allocated, images on CPU")

    # Free the GPU latents after decoding
    logger.info(f"  [DEBUG] Freeing samples...")
    del samples
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"  [DEBUG] After freeing samples: {allocated:.2f}GB allocated")

    return images


def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    import os  # Ensure os is available in local scope
    # Setup accelerator:
    device = accelerator.device

    # Seed all RNG streams from train.global_seed, identically on each rank so
    # all ranks build the same weights. Re-seeded per rank after
    # accelerator.prepare(). bf16 and SDPA kernels are non-deterministic, so
    # runs are not bit-reproducible.
    global_seed = train_config['train'].get('global_seed', 0)
    set_seed(global_seed)
    if accelerator.is_main_process:
        print(f"Seeded all RNG streams with train.global_seed={global_seed}")

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, accelerator.is_main_process)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Initialize WandB if requested (replaces TensorBoard)
        use_wandb = train_config['train'].get('use_wandb', False)
        if use_wandb:
            resume_training = train_config['train'].get('resume', False)
            wandb_utils.init(
                project=train_config['train'].get('wandb_project', 'lightningdit-brats'),
                entity=train_config['train'].get('wandb_entity', None),
                name=exp_name,
                config=train_config,
                resume=resume_training
            )
            logger.info(f"WandB initialized: project={train_config['train'].get('wandb_project', 'lightningdit-brats')}")
            logger.info(f"  Run name: {exp_name}")
            if resume_training:
                logger.info(f"  Resuming WandB run")
    else:
        use_wandb = False

    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio

    # Extract model config with new architectural features

    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    # Log architectural features
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("Model Architecture Configuration:")
        logger.info(f"  Model type: {train_config['model']['model_type']}")
        logger.info(f"  Input size: {latent_size}x{latent_size}")
        logger.info(f"  Hidden size: {model.hidden_size}")
        logger.info(f"  Depth: {model.depth}")
        logger.info(f"  Num heads: {model.num_heads}")
        logger.info(f"  Num classes: {train_config['data']['num_classes']}")
        logger.info(f"  In channels: {train_config['model'].get('in_chans', 4)}")
        logger.info("  Advanced features:")
        logger.info(f"    QKNorm: {train_config['model']['use_qknorm']}")
        logger.info(f"    SwiGLU: {train_config['model'].get('use_swiglu', False)}")
        logger.info(f"    RoPE: {train_config['model'].get('use_rope', False)}")
        logger.info(f"    RMSNorm: {train_config['model'].get('use_rmsnorm', False)}")
        logger.info(f"  Gradient checkpointing: {train_config['model'].get('use_checkpoint', False)}")
        logger.info("=" * 60)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train'] and train_config['train']['weight_init'] is not None:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        reuse_cfg = bool(train_config['train'].get('reuse_pretrained_cfg_token', False))
        model = load_weights_with_shape_check(model, checkpoint, rank=rank, reuse_pretrained_cfg_token=reuse_cfg)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank, reuse_pretrained_cfg_token=reuse_cfg)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")

    requires_grad(ema, False)

    # Note: Do NOT manually wrap with DDP - accelerate.prepare() handles this
    model = model.to(device)
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}, eps={train_config['optimizer'].get('epsilon', 1e-8)}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['optimizer']['lr'],
        weight_decay=0,
        betas=(0.9, train_config['optimizer']['beta2']),
        eps=float(train_config['optimizer'].get('epsilon', 1e-8)),
        )

    # Setup sharded data:
    if accelerator.is_main_process:
        logger.info("Setting up sharded latent datasets...")
        latent_norm = train_config['data'].get('latent_norm', False)
        latent_mult = train_config['data'].get('latent_multiplier', 1.0)
        logger.info(f"  Latent normalization: {latent_norm}")
        logger.info(f"  Latent multiplier: {latent_mult}")

    # Load data config (YAML file with dataset paths and settings)
    if 'data_config_path' not in train_config['data']:
        raise ValueError("data_config_path must be specified in train_config['data'] for sharded dataset")

    with open(train_config['data']['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)

    # Check for VA-VAE stats file if latent_norm is enabled
    if train_config['data'].get('latent_norm', False) and accelerator.is_main_process:
        vavae_stats_path = train_config['data'].get('vavae_stats_path', None)
        if vavae_stats_path:
            import os
            if os.path.exists(vavae_stats_path):
                logger.info(f"  Official VA-VAE stats found: {vavae_stats_path}")
                # Check if stats exist in dataset directories
                for ds_config in data_config['datasets']:
                    ds_stats = os.path.join(ds_config['path'], 'latents_stats.pt')
                    if not os.path.exists(ds_stats):
                        logger.warning(f"  ⚠️  latents_stats.pt not found in {ds_config['path']}")
                        logger.warning(f"     Consider copying from: {vavae_stats_path}")
                        logger.warning(f"     Stats will be auto-computed (may differ from official VA-VAE stats)")
            else:
                logger.warning(f"  ⚠️  Configured vavae_stats_path not found: {vavae_stats_path}")

    datasets = []
    vavae_stats_path = train_config['data'].get('vavae_stats_path', None)
    # Optional dataset subsetting. data.fraction in (0, 1]; default 1.0 = use
    # every shard. Selection is at the shard level, so the ShardWindowSampler
    # keeps working unchanged.
    dataset_fraction = float(train_config['data'].get('fraction', 1.0))
    if dataset_fraction <= 0.0 or dataset_fraction > 1.0:
        raise ValueError(f"data.fraction must be in (0, 1], got {dataset_fraction}")
    for ds_config in data_config['datasets']:
        ds_kwargs = dict(
            features_dir=ds_config['path'],
            shard_cache_size=ds_config.get('cache_size', 5),
            verbose=False,
            latent_norm=train_config['data'].get('latent_norm', False),
            latent_multiplier=train_config['data'].get('latent_multiplier', 1.0),
            vavae_stats_path=vavae_stats_path,
            dataset_fraction=dataset_fraction,
        )
        ds = ShardedLatentDataset(**ds_kwargs)
        datasets.append(ds)
        if accelerator.is_main_process:
            logger.info(
                f"  Loaded dataset: {ds_config['path']} ({len(ds):,} samples"
                + (f", fraction={dataset_fraction:.2f}" if dataset_fraction < 1.0 else "")
                + ")"
            )

    # Concatenate datasets if multiple
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
        if accelerator.is_main_process:
            logger.info(f"✓ Total samples: {len(dataset):,}")
    else:
        dataset = datasets[0]

    # Create shard window sampler for DDP
    rank = accelerator.local_process_index
    sampler = ShardWindowSampler(
        dataset=dataset,
        shard_window_size=data_config.get('shard_window_size', 15),
        num_replicas=accelerator.num_processes,
        rank=rank,
        shuffle=True,
        seed=train_config['train'].get('global_seed', 0)
    )

    # Get DataLoader config from YAML (with defaults)
    dataloader_config = data_config.get('dataloader', {})
    num_workers = dataloader_config.get('num_workers', train_config['data']['num_workers'])
    pin_memory = dataloader_config.get('pin_memory', True)
    drop_last = dataloader_config.get('drop_last', True)

    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes

    # Create DataLoader
    label_mode = train_config['data'].get('label_mode', 'has_tumor_extended')
    if label_mode == 'has_tumor_extended':
        # BraTS binary pathology appended on ImageNet (-1→1000,0→1001,1→1002)
        selected_collate = collate_has_tumor_extended
    elif label_mode == 'region_extended':
        # RadImageNet 5 regions appended on ImageNet (r→1001+r)
        selected_collate = collate_region_extended
    else:
        raise ValueError(f"Unknown data.label_mode: {label_mode}")
    if accelerator.is_main_process:
        logger.info(f"Using collate: {selected_collate.__name__} (label_mode='{label_mode}')")

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=selected_collate,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    if accelerator.is_main_process:
        logger.info(f"DataLoader ready: {len(dataset):,} samples, "
                   f"batch_size={batch_size_per_gpu}, num_workers={num_workers}")
        logger.info(f"Global batch size: {global_batch_size}")

    # Prepare models for training (following train.py order):
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    resumed = False
    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # Two ways to find the resume checkpoint:
        #   (a) train.resume_from = explicit path → load that exact file.
        #   (b) glob the local checkpoint_dir → legacy behavior, picks the
        #       largest file (the most recent ckpt of the current run).
        explicit_path = train_config['train'].get('resume_from')
        if explicit_path:
            checkpoint_files = [explicit_path] if os.path.isfile(explicit_path) else []
            if not checkpoint_files and accelerator.is_main_process:
                logger.warning(
                    f"resume_from={explicit_path} does not exist; "
                    f"falling back to glob({checkpoint_dir}/*.pt)."
                )
                checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        else:
            checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])
            # Restore AdamW optimizer state (m, v moments, per-param step counters)
            # so the resume continues with a true warm optimizer rather than
            # cold-start. opt is still un-wrapped here (prepare() runs below);
            # loading into the underlying optimizer is preserved through prepare.
            # Wrapped in try/except so an older ckpt that lacks an 'opt' key (or
            # an accelerate-wrapping incompat we haven't seen yet) doesn't crash
            # the run — falls back to cold optimizer + a logged warning.
            try:
                opt.load_state_dict(checkpoint['opt'])
                if accelerator.is_main_process:
                    logger.info("✓ Restored optimizer state from checkpoint")
            except Exception as e:
                if accelerator.is_main_process:
                    logger.warning(
                        f"Could not restore optimizer state ({e}); "
                        f"continuing with fresh optimizer (cold AdamW moments)."
                    )
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            resumed = True
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            # resume=True but no checkpoint yet — legitimate on the first run
            # with resume:true.
            # Must initialise train_steps here; the `if not resume` block below
            # won't fire, so without this it's UnboundLocalError at the loop.
            train_steps = 0
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")

    # Let accelerator prepare (wrap) model and optimizer only
    # NOTE: Don't prepare loader - we're using custom ShardWindowSampler for DDP
    model, opt = accelerator.prepare(model, opt)

    # Fork the RNG per rank (seed + process_index) so the noise / CFG-dropout
    # draws differ across GPUs. Model weights are already built and synced, so
    # diverging here is safe.
    set_seed(global_seed, device_specific=True)

    # Initialize EMA with synced weights (after model is wrapped)
    if not resumed:
        update_ema(ema, accelerator.unwrap_model(model), decay=0)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")
        logger.info(f"Training for {train_config['train']['max_steps']} steps...")

    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # Important for proper shuffling with sharded sampler
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")

        for batch in loader:
            x, y, _ = batch
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
            else:
                x = x.to(device)
                y = y.to(device)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, accelerator.unwrap_model(model))

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes using accelerate:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction="mean")
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss.item():.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Log to WandB (replaces TensorBoard)
                    if use_wandb:
                        wandb_utils.log({
                            "train_loss": avg_loss.item(),
                            "train_steps_per_sec": steps_per_sec
                        }, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Free checkpoint dict from CPU memory
                    del checkpoint

                # Aggressive GPU memory cleanup after checkpointing (all processes)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                accelerator.wait_for_everyone()

            # Generate visualization samples (if enabled) - SEPARATE from checkpoint saving
            # ONLY run on main process (GPU 0) to save memory
            sample_every = train_config['sample'].get('sample_every', 0)
            if sample_every > 0 and train_steps % sample_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    logger.info(f"Generating EMA samples at step {train_steps}...")

                    # Aggressively free up GPU memory before loading VAE
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Log memory status
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    logger.info(f"  GPU memory before VAE: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

                    # Load VAE for sampling (temporarily)
                    vae_name = train_config['vae']['model_name']
                    try:
                        vae = load_vae_for_sampling(vae_name, device, logger)

                        # Get sampling config
                        cfg_scale = train_config['sample'].get('cfg_scale', 1.5)
                        per_proc_batch_size = train_config['sample'].get('per_proc_batch_size', 4)

                        # Get model config
                        latent_channels = train_config['model'].get('in_chans', 32)
                        num_classes = train_config['data']['num_classes']
                        image_size = train_config['data']['image_size']

                        # Get latent normalization stats (for VA-VAE denormalization)
                        latent_mean = None
                        latent_std = None
                        latent_multiplier = train_config['data'].get('latent_multiplier', 1.0)
                        if train_config['data'].get('latent_norm', False):
                            # Access stats from first dataset
                            first_dataset = datasets[0] if len(datasets) > 0 else None
                            if first_dataset and hasattr(first_dataset, '_latent_mean'):
                                latent_mean = first_dataset._latent_mean
                                latent_std = first_dataset._latent_std
                                logger.info(f"  Using latent stats for denormalization: mean shape={latent_mean.shape}, std shape={latent_std.shape}")
                                logger.info(f"  Latent multiplier: {latent_multiplier}")

                        # Use fewer sampling steps for visualization to save memory
                        train_num_steps = train_config['sample'].get('train_num_sampling_steps', 50)
                        logger.info(f"  Using {train_num_steps} sampling steps for visualization (vs {train_config['sample']['num_sampling_steps']} for evaluation)")

                        # Generate samples
                        with torch.no_grad():
                            sample_images = generate_samples_for_visualization(
                                ema_model=ema,
                                transport=transport,
                                vae=vae,
                                vae_name=vae_name,
                                num_samples=per_proc_batch_size,
                                latent_size=latent_size,
                                latent_channels=latent_channels,
                                image_size=image_size,
                                num_classes=num_classes,
                                cfg_scale=cfg_scale,
                                device=device,
                                logger=logger,
                                latent_mean=latent_mean,
                                latent_std=latent_std,
                                latent_multiplier=latent_multiplier,
                                num_steps=train_num_steps,
                            )

                        # Log to WandB
                        if use_wandb:
                            wandb_utils.log_image(sample_images, step=train_steps)
                            logger.info(f"Generated {per_proc_batch_size} samples, logged to WandB")

                        # Free sample images from CPU memory
                        del sample_images

                        # Free VAE from GPU memory
                        del vae
                        torch.cuda.empty_cache()

                        # Aggressive GPU cleanup before resuming training
                        import gc
                        gc.collect()
                        torch.cuda.synchronize()

                        allocated = torch.cuda.memory_allocated(device) / 1024**3
                        reserved = torch.cuda.memory_reserved(device) / 1024**3
                        logger.info(f"  GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

                    except Exception as e:
                        logger.error(f"Error during sampling: {e}")
                        logger.error("Continuing training without samples...")

                accelerator.wait_for_everyone()

                # Evaluate on validation set (TODO: implement evaluate function if needed)
                # if 'valid_path' in train_config['data']:
                #     if accelerator.is_main_process:
                #         logger.info(f"Start evaluating at step {train_steps}")
                #     val_loss = evaluate(model, valid_loader, device, transport, (0.0, 1.0))
                #     dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                #     val_loss = val_loss.item() / dist.get_world_size()
                #     if accelerator.is_main_process:
                #         logger.info(f"Validation Loss: {val_loss:.4f}")
                #         if use_wandb:
                #             wandb_utils.log({"val_loss": val_loss}, step=train_steps)
                #     model.train()
            if train_steps >= train_config['train']['max_steps']:
                break
        epoch += 1  # Increment epoch for next iteration
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0, reuse_pretrained_cfg_token=False):
    import torch.nn.functional as F

    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # PatchEmbed conv with mismatched in_channels.
                # Cases this handles:
                #   - SD-VAE 4ch  -> VA-VAE 32ch
                # Copy the first min(ckpt, model) input channels. Extra channels stay zero.
                weight = torch.zeros_like(model_state_dict[name])
                ckpt_in = param.shape[1]
                model_in = model_state_dict[name].shape[1]
                n_copy = min(ckpt_in, model_in)
                weight[:, :n_copy] = param[:, :n_copy]
                model_state_dict[name] = weight
                if rank == 0:
                    print(
                        f"x_embedder.proj.weight: copied first {n_copy} of "
                        f"{model_in} input channels from ckpt ({ckpt_in}); "
                        f"remaining {max(0, model_in - n_copy)} channels zero-initialized."
                    )
            elif name == 'pos_embed':
                # Positional embeddings are deterministically computed using sin-cos functions
                # Skip loading and let model keep its recomputed values (recommended)
                old_seq_len = param.shape[1]
                new_seq_len = model_state_dict[name].shape[1]
                old_h = old_w = int(old_seq_len ** 0.5)
                new_h = new_w = int(new_seq_len ** 0.5)

                if rank == 0:
                    print(f"Skipping pos_embed: shape mismatch {param.shape} vs {model_state_dict[name].shape}")
                    print(f"  Positional embeddings will use model's sin-cos computed values for {new_h}x{new_w} grid")

                # # Alternative: Interpolate (commented out - use only if needed)
                # # checkpoint: [1, old_seq_len, embed_dim], model: [1, new_seq_len, embed_dim]
                # if rank == 0:
                #     print(f"Interpolating pos_embed from {param.shape} to {model_state_dict[name].shape}")
                # embed_dim = param.shape[2]
                # # Reshape to 2D spatial: [1, H, W, D] and permute to [1, D, H, W]
                # pos_embed_old = param.reshape(1, old_h, old_w, embed_dim).permute(0, 3, 1, 2)
                # # Interpolate using bilinear interpolation
                # pos_embed_new = F.interpolate(
                #     pos_embed_old,
                #     size=(new_h, new_w),
                #     mode='bilinear',
                #     align_corners=False
                # )
                # # Reshape back to [1, new_seq_len, embed_dim]
                # pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(1, new_seq_len, embed_dim)
                # model_state_dict[name].copy_(pos_embed_new)
                # if rank == 0:
                #     print(f"✓ Successfully interpolated pos_embed from {old_h}x{old_w} to {new_h}x{new_w}")

            elif name == 'y_embedder.embedding_table.weight':
                # Extend embedding table for new classes
                # checkpoint: [old_num_classes, embed_dim], model: [new_num_classes, embed_dim]
                old_num_classes = param.shape[0]
                new_num_classes = model_state_dict[name].shape[0]

                if rank == 0:
                    print(f"Extending embedding table from {old_num_classes} to {new_num_classes} classes")

                # Copy existing classes (ImageNet 1000 classes + CFG token)
                model_state_dict[name][:old_num_classes].copy_(param)

                # Initialize new classes (medical classes) with small random values
                # This matches the typical initialization for embedding tables
                new_classes = new_num_classes - old_num_classes
                torch.nn.init.normal_(model_state_dict[name][old_num_classes:], mean=0.0, std=0.02)
                if reuse_pretrained_cfg_token:
                    # Warm-init the new CFG-drop slot (last row of the extended table,
                    # = index num_classes used by LabelEmbedder) with the pretrained
                    # ImageNet CFG embedding (last row of the ckpt table). This lets
                    # the unconditional path reuse the learned ImageNet uncond token.
                    model_state_dict[name][-1].copy_(param[-1])
                    if rank == 0:
                        print(f"\u2713 Reused pretrained CFG token: ckpt row {old_num_classes-1} \u2192 model row {new_num_classes-1}")


                if rank == 0:
                    print(f"✓ Extended embedding table: copied {old_num_classes} classes, initialized {new_classes} new classes")

            elif name in ['feat_rope.freqs_cos', 'feat_rope.freqs_sin']:
                # RoPE frequencies are mathematically computed based on sequence length
                # Don't interpolate - skip loading and let model keep its recomputed values
                old_seq_len = param.shape[0]
                new_seq_len = model_state_dict[name].shape[0]

                if rank == 0:
                    print(f"Skipping {name}: shape mismatch {param.shape} vs {model_state_dict[name].shape}")
                    print(f"  RoPE frequencies will use model's computed values for seq_len={new_seq_len}")

            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)

    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir, is_main_process=True):
    """
    Create a logger that writes to a log file and stdout.
    Args:
        logging_dir: Directory to save log file
        is_main_process: Whether this is the main process (from accelerator)
    """
    if is_main_process:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    # Optional CLI overrides — let one config drive multiple runs that differ
    # only in (data fraction, init weights, exp name). When unset, the value
    # in the YAML config is used as-is.
    parser.add_argument('--data-fraction', type=float, default=None,
                        help='Override train_config["data"]["fraction"].')
    parser.add_argument('--weight-init', type=str, default=None,
                        help='Override train_config["train"]["weight_init"].')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Override train_config["train"]["exp_name"].')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override train_config["train"]["max_steps"].')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Override train_config["model"]["model_type"] '
                             '(e.g., LightningDiT-XL/1, LightningDiT-B/1).')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from a specific checkpoint path (e.g., '
                             'checkpoints/0050000.pt). When set, '
                             'overrides the glob-based local-dir search and '
                             'forces train.resume=True. The step counter is '
                             'parsed from the filename (e.g., 0050000.pt -> '
                             'train_steps=50000).')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    if args.data_fraction is not None:
        train_config['data']['fraction'] = args.data_fraction
    if args.weight_init is not None:
        # Sentinel: "none" / "null" / "" means "train from scratch" (don't load
        # any checkpoint). The training loop's existing
        #   if weight_init is not None: load_weights_with_shape_check(...)
        # branch then falls through and the model stays at its random init.
        if args.weight_init.lower() in ('none', 'null', ''):
            train_config['train']['weight_init'] = None
        else:
            train_config['train']['weight_init'] = args.weight_init
    if args.exp_name is not None:
        train_config['train']['exp_name'] = args.exp_name
    if args.max_steps is not None:
        train_config['train']['max_steps'] = args.max_steps
    if args.model_type is not None:
        train_config['model']['model_type'] = args.model_type
    if args.resume_from is not None:
        # Path-based resume: forces resume=True and passes the explicit ckpt
        # path to do_train, in place of the search in the local checkpoint dir.
        train_config['train']['resume'] = True
        train_config['train']['resume_from'] = args.resume_from
    do_train(train_config, accelerator)