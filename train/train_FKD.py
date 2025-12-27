import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import json
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR

from utils import AverageMeter, accuracy, get_parameters
sys.path.append('..')
from relabel.utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, RandomHorizontalFlipWithRes, mix_aug, load_weights_file

# Import custom parquet dataset for ImageNet-1K
try:
    from imagenet_parquet_dataset import ImageNetParquetDatasetFKD
    PARQUET_DATASET_AVAILABLE = True
except ImportError:
    PARQUET_DATASET_AVAILABLE = False
    print("Warning: imagenet_parquet_dataset not available. ImageNet-1K with parquet will not work.")

def collate_fn_fkd(batch):
    """Custom collate function for FKD dataloader"""
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    flip_status = [item[2] for item in batch]
    coords_status = torch.stack([item[3] for item in batch])
    weights = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    return images, targets, flip_status, coords_status, weights

# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md

from utils import save_arguments
from models.factory import ModelFactory



def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument("--dataset", default="imagenet", type=str,
                        choices=["imagenet", "tiny-imagenet", "imagenette", "cifar100"], help="dataset name")
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--input-size', default=224, type=int, metavar='S',
                        help='spatial size of input images')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')
    parser.add_argument("--exp-name", default="99", type=str, help="name of the experiment")

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')

    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')

    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--pseudo-label-csv', type=str, default=None,
                        help='path to CSV file with pseudo labels for training only')
    parser.add_argument('--use-parquet-dataset', action='store_true',
                        help='use parquet dataset for ImageNet-1K (no extraction needed)')
    parser.add_argument('--parquet-data-dir', type=str, default=None,
                        help='directory containing parquet files (if using parquet dataset)')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')

    args = parser.parse_args()

    # Set mode based on fkd_path
    if args.fkd_path is None or args.fkd_path == "dummy":
        args.mode = 'fkd_save'  # Use fkd_save mode to skip loading FKD data
    else:
        args.mode = 'fkd_load'
    return args

def main():
    args = get_args()
    script_name = os.path.basename(__file__)  # Get the script's filename
    save_arguments(script_name, args)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    # For parquet dataset, train_dir check is not needed
    if not args.use_parquet_dataset:
        assert os.path.exists(args.train_dir)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Check if using parquet dataset (for ImageNet-1K)
    use_parquet = args.use_parquet_dataset and args.dataset == 'imagenet' and args.pseudo_label_csv is not None
    
    if use_parquet:
        print("="*60)
        print("Using PARQUET DATASET for ImageNet-1K (no extraction needed)")
        print("="*60)
        
        if not PARQUET_DATASET_AVAILABLE:
            raise ImportError("imagenet_parquet_dataset module not found. Cannot use parquet dataset.")
        
        if args.parquet_data_dir is None:
            raise ValueError("--parquet-data-dir must be specified when using --use-parquet-dataset")
        
        # Create parquet dataset
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.08, 1), 
                                          interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = ImageNetParquetDatasetFKD(
            parquet_dir=args.parquet_data_dir,
            csv_path=args.pseudo_label_csv,
            label_column='pseudo_label_class_index',
            transform=train_transform,
            cache_parquet=False,  # Don't cache to avoid memory issues
            weights_map={},
            mode=args.mode,
            args_epoch=args.epochs if args.mode == 'fkd_load' else None,
            args_bs=args.batch_size if args.mode == 'fkd_load' else None
        )
        
        print(f"Parquet dataset loaded: {len(train_dataset)} images")
        
    else:
        # Original ImageFolder-based loading
        # load the weights of barycenters (if available)
        weight_file = os.path.join(args.train_dir, 'sample_weights.txt')
        if os.path.exists(weight_file):
            bary_weights_map = load_weights_file(args.train_dir, weight_file)
        else:
            # Create dummy weights map (all weights = 1.0) for real dataset training
            print("No sample_weights.txt found, using uniform weights (1.0 for all images)")
            bary_weights_map = {}

        train_dataset = ImageFolder_FKD_MIX(
            fkd_path=args.fkd_path if args.fkd_path != "dummy" else None,
            mode=args.mode,
            weights_map=bary_weights_map,
            args_epoch=args.epochs if args.mode == 'fkd_load' else None,
            args_bs=args.batch_size if args.mode == 'fkd_load' else None,
            root=args.train_dir,
            transform=ComposeWithCoords(transforms=[
                RandomResizedCropWithCoords(size=args.input_size,
                                            scale=(0.08, 1),
                                            interpolation=InterpolationMode.BILINEAR),
                RandomHorizontalFlipWithRes(),
                transforms.ToTensor(),
                normalize,
            ]))

        # If a CSV of pseudo-labels is provided, apply those labels to the training dataset only
        # (Skip this for parquet dataset as labels are already loaded)
        if args.pseudo_label_csv is not None:
            csv_path = args.pseudo_label_csv
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"pseudo-label CSV not found: {csv_path}")

            csv_map = {}
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    raw = row.get('image_path')
                    if raw is None:
                        continue
                    img_rel = raw.strip().strip('"').strip()
                    img_rel = img_rel.replace('\\', '/')
                    label_val = row.get('pseudo_label_class_index')
                    if label_val is None:
                        continue
                    lbl = int(label_val.strip().strip('"'))
                    csv_map[img_rel] = lbl

            # Apply mapping to train_dataset.samples
            new_samples = []
            new_targets = []
            unmatched = 0
            for (full_path, orig_target) in train_dataset.samples:
                rel = os.path.relpath(full_path, args.train_dir).replace('\\', '/')
                rel_candidates = [rel, './' + rel, rel.lstrip('./')]
                matched = False
                for key in rel_candidates:
                    if key in csv_map:
                        new_t = csv_map[key]
                        matched = True
                        break
                if not matched:
                    base = os.path.basename(full_path)
                    if base in csv_map:
                        new_t = csv_map[base]
                        matched = True
                if not matched:
                    new_t = orig_target
                    unmatched += 1

                new_samples.append((full_path, new_t))
                new_targets.append(new_t)

            train_dataset.samples = new_samples
            train_dataset.targets = new_targets
            print(f"Applied pseudo-label CSV: {csv_path}. Unmatched samples: {unmatched}")

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)

    # only main process, no worker process
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
    #     num_workers=0, pin_memory=True,
    #     prefetch_factor=None)
    # FKD loading requires single-process to properly handle batch-level mix_config
    # Multi-worker loading causes race conditions with shared batch state
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=0, pin_memory=True, collate_fn=collate_fn_fkd)

    # load validation data
    if use_parquet and args.val_dir is not None:
        # For parquet dataset, validation might also need special handling
        # Check if we have a validation CSV
        val_csv_path = args.pseudo_label_csv.replace('train_image_pseudo_labels', 'test_image_pseudo_labels')
        
        if os.path.exists(val_csv_path):
            print(f"Using parquet dataset for validation: {val_csv_path}")
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            
            from imagenet_parquet_dataset import ImageNetParquetDataset
            val_dataset = ImageNetParquetDataset(
                parquet_dir=args.parquet_data_dir,
                csv_path=val_csv_path,
                label_column='true_label_index',  # Use true labels for validation
                transform=val_transform,
                cache_parquet=False
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=int(args.batch_size/4), shuffle=False,
                num_workers=args.workers, pin_memory=True)
        else:
            print(f"Warning: Validation CSV not found at {val_csv_path}")
            print("Falling back to ImageFolder for validation (requires extracted images)")
            # Fallback to ImageFolder
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(args.val_dir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=int(args.batch_size/4), shuffle=False,
                num_workers=args.workers, pin_memory=True)
    elif args.input_size == 224:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=int(args.batch_size/4), shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.val_dir, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=int(args.batch_size / 4), shuffle=False,
            num_workers=args.workers, pin_memory=True)
    print('load data successfully')


    # load student model
    print("=> loading student model '{}'".format(args.model))

    num_class_map = {"imagenet": 1000, "tiny-imagenet": 200, "imagenette": 10, "cifar100": 100}
    assert args.dataset in num_class_map
    args.num_classes = num_class_map[args.dataset]

    # model = torchvision.models.__dict__[args.model](pretrained=False, num_classes=args.num_classes)
    # model = resnet18(args, num_classes=args.num_classes)
    model = ModelFactory.create(args.model, args, args.num_classes)
    if args.dataset == 'tiny-imagenet' and args.model.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.maxpool = nn.Identity()
    if args.dataset == 'cifar100' and args.model.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    model = nn.DataParallel(model).cuda()
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)


    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        # print(f"\nEpoch: {epoch}")

        train(model, args, epoch)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            top1 = validate(model, args, epoch)
        else:
            top1 = 0

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        """save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)"""

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='none')

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, (images, target, flip_status, coords_status, weights) in enumerate(args.train_loader):
        images = images.cuda()
        target = target.cuda()
        
        # Retrieve mix config that was loaded for this batch
        if hasattr(args.train_loader.dataset, 'last_mix_config') and args.train_loader.dataset.last_mix_config is not None:
            mix_index, mix_lam, mix_bbox, soft_label, batch_weights = args.train_loader.dataset.last_mix_config
            soft_label = soft_label.cuda().float()
        else:
            mix_index = mix_lam = mix_bbox = soft_label = batch_weights = None
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            
            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            if soft_label is not None:
                partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]
                output = F.log_softmax(output/args.temperature, dim=1)
                partial_soft_label = F.softmax(partial_soft_label/args.temperature, dim=1)
                loss = loss_function_kl(output, partial_soft_label)  # shape: (batch, num_classes)
                loss = torch.sum(loss, dim=1)  # shape: (batch,)
                # Multiply the loss terms with the weights
                if batch_weights is not None:
                    partial_weights = batch_weights[accum_id * small_bs: (accum_id + 1) * small_bs]
                    partial_weights = partial_weights / torch.mean(partial_weights)
                    loss = loss * partial_weights.to('cuda:0')
                loss = torch.mean(loss)  # scalar
            else:
                loss = F.cross_entropy(output, partial_target)
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        optimizer.step()



        # output = model(images)
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # output = F.log_softmax(output/args.temperature, dim=1)
        # soft_label = F.softmax(soft_label/args.temperature, dim=1)

        # loss = loss_function_kl(output, soft_label)
        # # loss = loss * args.temperature * args.temperature

        # n = images.size(0)
        # objs.update(loss.item(), n)
        # top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)

        # if batch_idx == 0:
        #     optimizer.zero_grad()

        # # do not support accumulate gradient, batch_size is fixed to 1024
        # assert args.gradient_accumulation_steps == 1
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps

        # loss.backward()

        # if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(args.train_loader) - 1:
        #     optimizer.step()
        #     optimizer.zero_grad()

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 acc = {:.6f},\t'.format(top1.avg) + \
                'Top-5 acc = {:.6f},\t'.format(top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    if epoch % 10 == 0:
        print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 acc = {:.6f},\t'.format(top1.avg) + \
              'Top-5 acc = {:.6f},\t'.format(top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    result_dict = {'acc': round(top1.avg, 6)}
    filename = f"../log/{args.exp_name}.json"

    # Save the result.
    with open(filename, 'r') as f:
        data = json.load(f)

    data.append(result_dict)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    return top1.avg

def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()