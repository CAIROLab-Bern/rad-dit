import math
import sys
import os

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
from util import wandb_utils as wbu
import copy

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        if getattr(args, 'dataset', '') == 'brats_gli_multichannel':
            # BraTS safetensors already hold fp32 in [-1, 1] (no PNG round-trip), so
            # the legacy /255 + *2-1 conversion would squish input to ~-1 everywhere.
            x = x.to(device, non_blocking=True).to(torch.float32)
        else:
            x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
            x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        global_step = epoch * len(data_loader) + data_iter_step

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                # Peak VRAM since last reset: useful for batch-size sweeps.
                peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                log_writer.add_scalar('vram/peak_alloc_gb', peak_alloc_gb, epoch_1000x)
                log_writer.add_scalar('vram/peak_reserved_gb', peak_reserved_gb, epoch_1000x)
                wandb_stats = {
                    'train_loss': loss_value_reduce,
                    'lr': lr,
                    'epoch': epoch,
                    'vram/peak_alloc_gb': peak_alloc_gb,
                    'vram/peak_reserved_gb': peak_reserved_gb,
                }
                wbu.log(wandb_stats, step=global_step)
                torch.cuda.reset_peak_memory_stats()

        if getattr(args, 'max_steps', 0) and data_iter_step + 1 >= args.max_steps:
            print(f"[smoke test] Reached max_steps={args.max_steps}, breaking epoch early")
            break


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    # For "extended" label modes (RadImageNet on top of an ImageNet-pretrained
    # ckpt) only labels label_offset..label_offset+effective_classes-1 are
    # meaningful, even though the embedder table has class_num+1 rows. The
    # plain ImageNet path keeps the original behaviour (class_num=1000, eval
    # over arange(1000)) since label_offset defaults to 0 and we fall back to
    # class_num.
    label_offset = getattr(args, 'label_offset', 0)
    effective_classes = getattr(args, 'eval_class_num', None) or args.class_num
    assert args.num_images % effective_classes == 0, \
        "Number of images per class must be the same"
    class_label_gen_world = np.arange(
        label_offset, label_offset + effective_classes
    ).repeat(args.num_images // effective_classes)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        # For 3-channel RGB outputs: write a single BGR PNG (legacy path).
        # For 4/5-channel BraTS outputs: save one PNG per channel as 3-channel
        # grayscale-stored-as-RGB (replicate single channel three times) so the
        # PNGs match the reference images.
        # That lets the same dgm-eval `--avg-all-channels` flag work for both
        # the legacy RadIN pipeline and BraTS per-modality / flat FID.
        # Extra rule for n_ch==5 (BraTS 4 MRI + seg): if the sample's label is
        # the "no_tumor" row (label_offset), force ch4 (seg) to all-black so
        # downstream FID/visualisation never gets a meaningless predicted mask.
        n_ch = sampled_images.size(1)
        no_tumor_label = getattr(args, 'label_offset', 0)
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            arr = np.round(np.clip(sampled_images[b_id].numpy() * 255, 0, 255)).astype(np.uint8)
            if n_ch == 3:
                # legacy path: HxWxC then BGR-reverse, single PNG
                gen_img = arr.transpose([1, 2, 0])[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)
            else:
                label_b = int(labels_gen[b_id].item())
                if getattr(args, 'save_random_channel', False):
                    # --save_random_channel: save ONE random channel per generation
                    # event. This produces statistically independent single-modality
                    # samples for fair comparison vs single-channel models. The
                    # img_id seeds the RNG so output is reproducible across runs.
                    valid_channels = [c for c in range(n_ch)
                                      if not (c == 4 and label_b == no_tumor_label)]
                    rng = np.random.default_rng(img_id)
                    chosen_c = int(rng.choice(valid_channels))
                    arr_c = arr[chosen_c]
                    gen_img = np.stack([arr_c, arr_c, arr_c], axis=-1)
                    cv2.imwrite(
                        os.path.join(save_folder, '{}_ch{}.png'.format(str(img_id).zfill(5), chosen_c)),
                        gen_img,
                    )
                else:
                    for c in range(n_ch):
                        if c == 4 and label_b == no_tumor_label:
                            arr_c = np.zeros_like(arr[c])
                        else:
                            arr_c = arr[c]
                        gen_img = np.stack([arr_c, arr_c, arr_c], axis=-1)   # [H, W, 3] gray-as-RGB
                        cv2.imwrite(
                            os.path.join(save_folder, '{}_ch{}.png'.format(str(img_id).zfill(5), c)),
                            gen_img,
                        )

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)


    torch.distributed.barrier()
