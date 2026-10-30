import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.radimagenet import RadImageNetDataset

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from util.pretrained import load_jit_pretrained


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--save_random_channel', action='store_true',
                        help='For multichannel BraTS gen: save ONE random channel per '
                             'generation event (vs all). Produces statistically independent '
                             'single-modality PNGs for fair comparison against single-channel '
                             'models. Use with --num_images=50000 to match single-channel FID.')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--eval_class_num', default=0, type=int,
                        help='Number of classes to sweep over during eval (defaults to class_num). '
                             'For extended-label runs set this to 5 (regions) so eval doesn\'t '
                             'sample all 1005 rows.')
    parser.add_argument('--dataset', default='imagenet',
                        choices=['imagenet', 'radimagenet', 'radimagenet_pathology',
                                 'brats_gli_multichannel', 'mixed_radnat'],
                        help='Which dataset/loader to use. brats_gli_multichannel reads '
                             'sharded safetensors with N-channel slices (4ch=t1n+t1c+t2w+t2f, '
                             '5ch adds seg_mask). Pair with --in_channels {4,5}. '
                             'mixed_radnat = MixedRadNat (per-step probabilistic mix of RadIN + ImageNet; '
                             'requires --nat_data_path and --p_nat).')
    parser.add_argument('--in_channels', default=3, type=int,
                        help='Number of input/output channels for the JiT denoiser. '
                             'Default 3 (RGB ImageNet/RadIN); pass 4 or 5 for BraTS multichannel. '
                             'When using --pretrained_ckpt with a different channel count, the '
                             'PatchEmbed/FinalLayer weights are auto-grown via avg+3/N-rescale.')
    parser.add_argument('--brats_shard_dir', default='', type=str,
                        help='Path to sharded safetensors train dir for '
                             '--dataset brats_gli_multichannel (e.g. '
                             'data/brats24_gli_multichannel_256/train_shards/).')
    parser.add_argument('--select_channel', default='', type=str,
                        help='When --in_channels 1, pick one BraTS modality '
                             '(one of t1n/t1c/t2w/t2f).')
    parser.add_argument('--brats_flat', action='store_true',
                        help='Flat single-channel BraTS: cycle all 4 modalities, '
                             'dataset size = 4× base slices, replicate selected '
                             'modality across 3 RGB channels. Requires '
                             '--in_channels 3 and no --select_channel. Matches '
                             'LDiT single-channel BraTS setup.')
    parser.add_argument('--nat_data_path', default='', type=str,
                        help='ImageNet root for --dataset mixed_radnat. '
                             'Expected layout: <nat_data_path>/train/<class>/*.JPEG '
                             '(standard torchvision.datasets.ImageFolder).')
    parser.add_argument('--p_nat', default=0.10, type=float,
                        help='Per-step probability of drawing a natural-image sample '
                             '(ImageNet) in --dataset mixed_radnat. Default 0.10. '
                             'The complement (1-p_nat) is drawn from RadIN.')
    parser.add_argument('--label_mode', default='plain', choices=['plain', 'extended'],
                        help='plain: labels 0..class_num-1. '
                             'extended: shift labels by --label_offset so ImageNet 0-999 + CFG-1000 '
                             'stay preserved (used when fine-tuning from an ImageNet-pretrained ckpt).')
    parser.add_argument('--label_offset', default=0, type=int,
                        help='Offset added to dataset labels (e.g. 1001 for RadImageNet 5 regions '
                             'appended on top of ImageNet 0-999 + CFG-1000).')
    parser.add_argument('--cfg_drop_index', default=-1, type=int,
                        help='Embedding row used as the CFG-drop / uncond token. '
                             'Default -1 = use num_classes (legacy upstream behaviour: '
                             'drop sits at the LAST row of the +1 embedding table). '
                             'For extended-pretrained runs (--class_num 1005 --label_offset 1001), '
                             'pass 1000 to reuse the preserved ImageNet CFG token at row 1000 '
                             'and isolate region 4 (which otherwise collides at row 1005).')
    parser.add_argument('--pretrained_ckpt', default='', type=str,
                        help='Path to a pretrained JiT checkpoint to initialise from. '
                             'Only used when --resume does not yield a checkpoint-last.pth.')
    parser.add_argument('--exclude_paths', default='', type=str,
                        help='Optional text file with validation images to exclude, one path per line. '
                             'Match is by relative path.')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoint-last.pth (overwriting)')
    parser.add_argument('--save_persistent_freq', type=int, default=100,
                        help='Frequency (in epochs) to save persistent (epoch-named, non-overwriting) checkpoints. '
                             'Default 100 preserves the original ImageNet 600-epoch schedule. '
                             'For shorter runs (e.g. 80 epochs on RadImageNet), set this lower (e.g. 20).')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--max_steps', default=0, type=int,
                        help='If >0, break training after this many optimizer steps per epoch '
                             '(used for smoke tests). 0 = full epoch.')

    # wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Mirror train_loss / lr to wandb (in addition to TensorBoard).')
    parser.add_argument('--wandb_project', default='radimagenet-jit', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default='', type=str,
                        help='Defaults to basename of --output_dir.')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    # The CFG row must not be a class label.
    cfg_row = args.cfg_drop_index if args.cfg_drop_index >= 0 else args.class_num
    last_label = args.class_num if args.label_mode == 'extended' else args.class_num - 1
    if args.label_offset <= cfg_row <= last_label:
        raise ValueError(f"CFG row {cfg_row} is also a class label ({args.label_offset}..{last_label}). "
                         f"With an ImageNet start, pass --cfg_drop_index 1000.")

    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.use_wandb:
        from util import wandb_utils as wbu
        wbu.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(os.path.normpath(args.output_dir)),
            config=args,
            resume=bool(args.resume) and os.path.exists(
                os.path.join(args.resume, "checkpoint-last.pth")),
        )

    # Generating images needs only a checkpoint, so skip the training data.
    if not args.evaluate_gen:
        # Data augmentation transforms
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ])

        if args.dataset == 'imagenet':
            dataset_train = datasets.ImageFolder(
                os.path.join(args.data_path, 'train'), transform=transform_train
            )
        elif args.dataset == 'radimagenet':
            dataset_train = RadImageNetDataset(
                args.data_path, transform=transform_train, label_offset=args.label_offset,
                exclude_paths=args.exclude_paths or None,
            )
        elif args.dataset == 'brats_gli_multichannel':
            # Multichannel BraTS — reads sharded safetensors with [N, 4 or 5, 256, 256] slices.
            # No PIL transforms (data already in [-1, 1] float and pre-upsampled to 256).
            from util.brats_multichannel import MultiChannelBraTSDataset
            shard_dir = args.brats_shard_dir or args.data_path
            dataset_train = MultiChannelBraTSDataset(
                shard_dir=shard_dir, in_channels=args.in_channels, label_offset=args.label_offset,
                select_channel=args.select_channel or None,
                flat_mode=args.brats_flat,
            )
        elif args.dataset == 'mixed_radnat':
            # Per-step probabilistic mix of RadIN + ImageNet. Used for the mixed-task
            # training that prevents catastrophic forgetting of ImageNet during RadIN
            # fine-tune. Shared y_embedder (1006 rows: ImNet 0..999, CFG-1000, RadIN
            # 1001..1005). Pair with --label_mode extended --label_offset 1001
            # --class_num 1005 --cfg_drop_index 1000.
            from util.mixed_radnat import MixedRadNat
            assert args.nat_data_path, '--dataset mixed_radnat requires --nat_data_path'
            nat_root = os.path.join(args.nat_data_path, 'train')
            dataset_train = MixedRadNat(
                rad_root=args.data_path, nat_root=nat_root,
                transform=transform_train, p_nat=args.p_nat,
                rad_label_offset=args.label_offset,
                rad_exclude_paths=args.exclude_paths or None,
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train =", sampler_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            # Eval takes no optimizer steps, so skip the optimizer state.
            if not getattr(args, 'evaluate_gen', False):
                optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    elif args.evaluate_gen:
        raise FileNotFoundError(f"--evaluate_gen needs a checkpoint: {checkpoint_path} does not exist")
    elif args.pretrained_ckpt:
        if not os.path.exists(args.pretrained_ckpt):
            raise FileNotFoundError(f"--pretrained_ckpt does not exist: {args.pretrained_ckpt}")
        load_jit_pretrained(model_without_ddp, args.pretrained_ckpt, args)
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Initialised from pretrained ckpt:", args.pretrained_ckpt)
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        persistent_save = (epoch % args.save_persistent_freq == 0 and epoch > 0) \
            or (epoch + 1 == args.epochs)
        if persistent_save:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
