import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

# import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import (
    count_parameters, init_distributed_mode, get_rank, get_world_size,
    is_main_process, load_model
)


import models_mmae
import models_mmae_teacher

from engine import pretrain_one_epoch


from contextlib import suppress
from util.pos_embed import interpolate_pos_embed




def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_args_parser():
    parser = argparse.ArgumentParser('flow mamba pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--steps', default=150000, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=10000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='net_mamba_pretrain', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_teacher', default='net_mamba_classifier', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--ema_op', default='per_epoch', type=str)
    parser.add_argument('--ema_frequent', default=1, type=int, help='how frequent the ema do')

    parser.add_argument('--model_teacher_path', default=None, help='the path of teachers checkpoint')
    parser.add_argument('--momentum_teacher', default=0.96, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to [momentum_teacher_final] during training with cosine schedule.
            We recommend setting a higher value with small batches: for example use 0.96 to 0.99 with batch size of 256.""")
    parser.add_argument('--momentum_teacher_warmup_ep', default=0, type=int,
                        help="""Number of momentum warmup epochs""")
    parser.add_argument('--momentum_teacher_warmup', default=0.0, type=float, help="""Only worked when momentum_teacher_warmup_ep > 0, The EMA
            parameter for teacher update. The value is increased from [momentum_teacher_warmup] to [momentum_teacher] in the first [momentum_teacher_warmup_ep] epochs""")
    parser.add_argument('--momentum_teacher_final', default=0.99, type=float, help="""The end value of base EMA
            parameter for teacher update. We recommend setting a higher value with small batches: for example use 0.96 with batch size of 256.""")

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=25, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    # pretrain tasks
    parser.add_argument('--pop', action='store_true', help='packet order prediction')
    parser.add_argument('--pop_loss_weight', default=0.01, type=float,
                        help='packet order prediction loss weight')

    parser.add_argument('--learning_loss:', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(learning_loss=True)


    return parser


def normalize_array(tensor, mean, std, dtype=torch.float32):
    # Ensure mean and std are tensors and on the same device as the input tensor
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    # Check if std contains any zero values to prevent division by zero
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    # Reshape mean and std to match the tensor shape if they are 1-dimensional
    if mean.ndim == 1:
        mean = mean.view(-1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1)

    return tensor.sub_(mean).div_(std)

def min_max_normalize(tensor):
    return (tensor) / (255)

class NPYPipelineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._make_dataset()
        self.label_to_idx = self._get_label_to_idx()


    def _make_dataset(self):
        samples = []
        for root, _, fnames in sorted(os.walk(self.root_dir)):
            for fname in sorted(fnames):
                if fname.endswith('.npy'):
                    path = os.path.join(root, fname)
                    label = os.path.basename(os.path.dirname(path))
                    samples.append((path, label))
        if not samples:
            raise RuntimeError(f"No .npy files found in {self.root_dir}")
        return samples

    def _get_label_to_idx(self):
        labels = sorted(set(label for _, label in self.samples))
        if not labels:
            raise RuntimeError("No labels found. Check your dataset structure.")
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        data = min_max_normalize(data)

        if self.transform is not None:

            mean = [0.5]
            std = [0.5]
            data = normalize_array(data, mean, std)

        label_idx = self.label_to_idx[label]
        return data, label_idx



def main(args):
    init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True



    class ToTensor(object):
        def __call__(self, x):
            return torch.tensor(x, dtype=torch.float32)  # Add channel dimension

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, x):
            return (x - self.mean) / self.std

    transform_train = transforms.Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
    ])



    dataset = NPYPipelineDataset(os.path.join(args.data_path), transform=transform_train)
    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    print("Sampler_train = %s" % str(sampler))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    epochs = int(args.steps / len(data_loader_train)) + 1
    args.epochs = epochs
    # define the model
    momentum_schedule = cosine_scheduler(base_value=args.momentum_teacher,
                                         final_value=args.momentum_teacher_final,
                                         epochs=args.epochs,
                                         niter_per_ep=1,
                                         warmup_epochs=args.momentum_teacher_warmup_ep,
                                         start_warmup_value=args.momentum_teacher_warmup)

    model = models_mmae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
    )
    model_teacher = models_mmae_teacher.__dict__[args.model_teacher](
        norm_pix_loss=args.norm_pix_loss,

    )

    model.to(device)
    for n, p in model.named_parameters():
        print(n)


    if args.model_teacher_path:
        checkpoint = torch.load(args.model_teacher_path, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.model_teacher_path)
        checkpoint_model = checkpoint['model']
        state_dict = model_teacher.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model_teacher, checkpoint_model)

        # load pre-trained model
        msg = model_teacher.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)

    model_teacher.to(device)

    model_without_ddp = model
    model_teacher_without_ddp = model_teacher
    print("Model = %s" % str(model_without_ddp))
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model_teacher_without_ddp = model_teacher.module


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp,
    optimizer=optimizer,
    loss_scaler=loss_scaler,
    model_teacher_without_ddp=model_teacher_without_ddp)


    print(f"Start training for {args.steps} steps")
    start_time = time.time()

    loss_pred = None
    for epoch in range(0, epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, loss_pred_s = pretrain_one_epoch(
            model, model_teacher, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            log_writer=log_writer,
            model_without_ddp=model_without_ddp,
            momentum_schedule=momentum_schedule,
            ema_op=args.ema_op, is_ema=True if (epoch + 1) % args.ema_frequent == 0 else False,
            args=args,
            loss_pred_s=loss_pred,
        )
        loss_pred = loss_pred_s
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    with open(os.path.join(args.output_dir, "train_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "total_time": total_time,
        }, f, indent=2)

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


