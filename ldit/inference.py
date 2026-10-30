"""
Sampling Scripts of LightningDiT.

by Maple (Jingfeng Yao) from HUST-VL
"""

import os, math, json, pickle, logging, argparse, yaml, torch, numpy as np
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
# local imports
import wandb_utils
from tokenizer.vavae import VA_VAE
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from datasets.img_latent_dataset import ImgLatentDataset
from train_shards import load_weights_with_shape_check

# Helper function for parallel image saving
def save_image_worker(image_data):
    """Worker function to save a single image in parallel"""
    sample, filepath = image_data
    Image.fromarray(sample).save(filepath)
    return filepath

# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None):
    """
    Run sampling.
    """

    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

    sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
    if accelerator.process_index == 0:
        print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    if not os.path.exists(sample_folder_dir):
        if accelerator.process_index == 0:
            os.makedirs(sample_folder_dir, exist_ok=True) 
    else:
        png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
        png_count = len(png_files)
        if png_count >= train_config['sample']['fid_num']:
            if accelerator.process_index == 0:
                print_with_prefix(f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling.")
            return sample_folder_dir

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    latent_size = train_config['data']['image_size'] // downsample_ratio

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    # Prepare checkpoint in format expected by load_weights_with_shape_check
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint_dict = {"model": checkpoint["ema"]}
    elif "model" in checkpoint:
        checkpoint_dict = checkpoint
    else:
        # Raw state dict
        checkpoint_dict = {"model": checkpoint}

    # Use shape-aware loading to handle different resolutions and class counts
    model = load_weights_with_shape_check(model, checkpoint_dict, rank=accelerator.local_process_index)
    model.eval()  # important!
    model.to(device)

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity;
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    if vae is None:
        vae = VA_VAE(
            f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml',
        )
        if accelerator.process_index == 0:
            print_with_prefix('Loaded VAE model')

    using_cfg = cfg_scale > 1.0
    if using_cfg:
        if accelerator.process_index == 0:
            print_with_prefix('Using cfg:', using_cfg)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    if rank == 0:
        if accelerator.process_index == 0:
            print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // accelerator.num_processes) // n)
    # Continue after the finished batches. A new seed stops the restart from repeating images.
    done_iterations = min(done_iterations, iterations)
    if done_iterations > 0:
        torch.manual_seed(torch.initial_seed() + done_iterations)
        if accelerator.process_index == 0:
            print_with_prefix(f"Found {num_samples} PNG files, continuing from batch {done_iterations}.")
    pbar = range(done_iterations, iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = done_iterations * global_batch_size
    
    if accelerator.process_index == 0:
        print_with_prefix("Using latent normalization")
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)


    for i in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)

        # Sample class labels: uniform by default, or data.class_probs if set.
        # Medical classes start at 1001. The CFG slot at num_classes_cfg is excluded.
        class_probs = train_config['data'].get('class_probs', None)
        num_classes_cfg = train_config['data']['num_classes']
        if class_probs is not None:
            if len(class_probs) != num_classes_cfg - 1001:
                raise ValueError(f"class_probs needs {num_classes_cfg - 1001} values, got {len(class_probs)}")
            probs = torch.tensor(class_probs, device=device, dtype=torch.float32)
            y = 1001 + torch.multinomial(probs, n, replacement=True)
        elif num_classes_cfg > 1000:
            y = torch.randint(1001, num_classes_cfg, (n,), device=device)
        else:
            # ImageNet-only model
            y = torch.randint(0, num_classes_cfg, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            # CFG-null token: the row that training drops labels to (index num_classes).
            y_null = torch.full((n,), num_classes_cfg, device=device, dtype=torch.long)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = (samples * latent_std) / latent_multiplier + latent_mean
        samples = vae.decode_to_images(samples)

        # Save samples to disk as individual .png files (parallel I/O)
        # Prepare save tasks
        save_tasks = []
        for i, sample in enumerate(samples):
            index = i * accelerator.num_processes + accelerator.process_index + total
            filepath = f"{sample_folder_dir}/{index:06d}.png"
            save_tasks.append((sample, filepath))

        # Execute saves in parallel (32 worker threads for PNG encoding)
        with ThreadPoolExecutor(max_workers=32) as executor:
            list(executor.map(save_image_worker, save_tasks))

        total += global_batch_size
        accelerator.wait_for_everyone()

    return sample_folder_dir

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory from config')
    parser.add_argument('--num-samples', type=int, default=None, help='Override number of samples from config')
    parser.add_argument('--cfg-scale', type=float, default=None, help='Override CFG scale from config')
    parser.add_argument('--num-sampling-steps', type=int, default=None, help='Override sampling steps from config')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed from config')
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    # Apply CLI overrides
    train_config['ckpt_path'] = args.ckpt
    if args.output_dir is not None:
        train_config['train']['output_dir'] = args.output_dir
    if args.num_samples is not None:
        train_config['sample']['fid_num'] = args.num_samples
    if args.cfg_scale is not None:
        train_config['sample']['cfg_scale'] = args.cfg_scale
    if args.num_sampling_steps is not None:
        train_config['sample']['num_sampling_steps'] = args.num_sampling_steps
    if args.seed is not None:
        train_config['train']['global_seed'] = args.seed

    # Initialize WandB (only on main process)
    if accelerator.is_main_process and train_config['train'].get('use_wandb', False):
        try:
            wandb_utils.init(
                project=train_config['train'].get('wandb_project', 'lightningdit-inference'),
                entity=train_config['train'].get('wandb_entity', None),
                name=train_config['train']['exp_name'],
                config=train_config
            )
        except Exception as e:
            print_with_prefix(f"WandB initialization failed: {e}")
            print_with_prefix("Continuing without WandB logging...")

    # get ckpt_dir
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16

    # get model
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        learn_sigma=train_config['model']['learn_sigma'] if 'learn_sigma' in train_config['model'] else False,
    )

    # naive sample
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model)
    
    if accelerator.process_index == 0:
        print_with_prefix(f'Generated samples saved to: {sample_folder_dir}')
        print_with_prefix('Use dgm_eval for FID calculation')
