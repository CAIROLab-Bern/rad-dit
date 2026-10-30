import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import sys

import numpy as np
import torch
from PIL import Image
from .dataloaders import get_dataloader
from .metrics import *
from .models import load_encoder, MODELS
from .representations import get_representations, load_reps_from_path, save_outputs


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default='inception', choices=MODELS.keys(),
                    help='Model to use for generating feature representations.')


parser.add_argument('-bs', '--batch_size', type=int, default=50,
                    help='Batch size to use')

parser.add_argument('--num-workers', type=int,
                    help='Number of processes to use for data loading. '
                         'Defaults to `min(8, num_cpus)`')

parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

parser.add_argument('--nearest_k', type=int, default=5,
                    help='Number of neighbours for precision, recall, density, and coverage')

parser.add_argument('--reduced_n', type=int, default=10000,
                    help='Number of samples used for train, baseline, test, and generated sets for FLS')

parser.add_argument('--nsample', type=int, default=50000,
                    help='Maximum number of images to use for calculation')

parser.add_argument('path', type=str, nargs='+',
                    help='Paths to the images, the first one is the real dataset, followed by generated')


parser.add_argument('--metrics', type=str, nargs='+', default=['fd', 'kd', 'prdc'],
                    help='Metrics to compute: fd, kd, prdc')






parser.add_argument('--output_dir', type=str, default='experiments/',
                    help='Directory to save outputs in')

parser.add_argument('--save', action='store_true',
                    help='Save representations to output_dir')

parser.add_argument('--load', action='store_true',
                    help='Load representations and statistics from previous runs if possible')

parser.add_argument('--no-load', action='store_false', dest='load',
                    help='Do not load representations and statistics from previous runs.')
parser.set_defaults(load=True)

parser.add_argument('--seed', type=int, default=13579,
                    help='Random seed')

parser.add_argument('--clean_resize', action='store_true',
                    help='Use clean resizing (from pillow)')


parser.add_argument('--avg-all-channels', action='store_true',
                    help='Replace all channels with mean(ch0, ch1, ch2) before feature extraction. '
                         'Use for images where all 3 channels carry signal (e.g. (T,T,T) grayscale).')

parser.add_argument('--depth', type=int, default=0,
                    help='Negative depth for internal layers, positive 1 for after projection head.')


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        try:
            # os. generally works on unix systems
            num_avail_cpus = len(os.sched_getaffinity(0))
        except:
            # Windows or other systems do not support os.sched_getaffinity. 
            # And dataloader can have issues with num_workers>0.
            # https://discuss.pytorch.org/t/eoferror-ran-out-of-input-when-enumerating-the-train-loader/22692/7
            print(f'\nWarning: os.sched_getaffinity error. Limited number of CPUs/workers used.\n', file=sys.stderr)
            num_avail_cpus = 0
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers

    return device, num_workers


def get_dataloader_from_path(path, model_transform, num_workers, args, sample_w_replacement=False):
    print(f'Getting DataLoader for path: {path}\n', file=sys.stderr)

    if getattr(args, 'avg_all_channels', False):
        def transform(x):
            arr = np.array(x)
            mean_ch = (arr[:, :, 0].astype(np.uint16) + arr[:, :, 1].astype(np.uint16) + arr[:, :, 2].astype(np.uint16)) // 3
            arr[:, :, 0] = mean_ch
            arr[:, :, 1] = mean_ch
            arr[:, :, 2] = mean_ch
            return model_transform(Image.fromarray(arr))
    else:
        transform = lambda x: model_transform(x)

    dataloader = get_dataloader(path, args.nsample, args.batch_size, num_workers, seed=args.seed,
                                sample_w_replacement=sample_w_replacement,
                                transform=transform)

    return dataloader


def compute_representations(DL, model, device, args):
    """
    Load representations from disk if path exists,
    else compute image representations using the specified encoder

    Returns:
        repsi: float32 (Nimage, ndim)
    """

    if args.load:
        print(f'Loading saved representations from: {args.output_dir}\n', file=sys.stderr)
        repsi = load_reps_from_path(args.output_dir, args.model, None, DL)
        if repsi is not None: 
            return repsi

        print(f'No saved representations found: {args.output_dir}\n', file=sys.stderr)

    print('Calculating Representations\n', file=sys.stderr)
    repsi = get_representations(model, DL, device, normalized=False)
    if args.save:
        print(f'Saving representations to {args.output_dir}\n', file=sys.stderr)
        save_outputs(args.output_dir, repsi, args.model, None, DL)
    return repsi


def compute_scores(args, reps):

    scores = {}

    if 'fd' in args.metrics:
        print("Computing FD \n", file=sys.stderr)
        scores['fd'] = compute_FD_with_reps(*reps)

    if 'kd' in args.metrics:
        print("Computing KD \n", file=sys.stderr)
        mmd_values = compute_mmd(*reps)
        scores['kd_value'] = mmd_values.mean()
        scores['kd_variance'] = mmd_values.std()

    if 'prdc' in args.metrics:
        print("Computing precision, recall, density, and coverage \n", file=sys.stderr)
        reduced_n = min(args.reduced_n, reps[0].shape[0], reps[1].shape[0])
        inds0 = np.random.choice(reps[0].shape[0], reduced_n, replace=False)
        inds1 = np.random.choice(reps[1].shape[0], reduced_n, replace=False)
        prdc_dict = compute_prdc(reps[0][inds0], reps[1][inds1], nearest_k=args.nearest_k)
        scores = dict(scores, **prdc_dict)

    for key, value in scores.items():
        print(f'{key}: {value:.5f}\n')

    return scores


def save_scores(scores, args):

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.model}_scores_nimage-{args.nsample}")
    np.savez(f'{out_path}.npz', scores=scores, run_params=vars(args))


def main():
    args = parser.parse_args()
    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)

    print('Loading Model', file=sys.stderr)
    model = load_encoder(args.model, device, ckpt=None, arch=None,
                         clean_resize=args.clean_resize,
                         sinception=False,
                         depth=args.depth)
    dataloader_real = get_dataloader_from_path(args.path[0], model.transform, num_workers, args)
    reps_real = compute_representations(dataloader_real, model, device, args)

    # Loop over all generated paths
    all_scores = {}
    for i, path in enumerate(args.path[1:]):
        dataloaderi = get_dataloader_from_path(path, model.transform, num_workers, args)
        repsi = compute_representations(dataloaderi, model, device, args)
        print(f'Computing scores between reference dataset and {path}\n', file=sys.stderr)
        all_scores[f'run{i:02d}'] = compute_scores(args, [reps_real, repsi])

    save_scores(all_scores, args)


if __name__ == '__main__':
    main()
