import torch.nn as nn
import torchvision
import os
import yaml
import torchvision.transforms as transforms
from criterions import LabelSmoothingCrossEntropyLoss
from da import RandomCropPaste

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, "config.yaml"), "w") as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'vit':
        from vit import ViT
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            attn_type=args.attn_type
            )
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    if args.dataset in ['c10','c100','svhn']:
        train_transform += [
            transforms.RandomCrop(size=args.size, padding=args.padding)
        ]
    elif args.dataset == 'imagenet':
        train_transform += [
            transforms.RandomResizedCrop(size=args.size, scale=(0.08, 1.0))
        ]
    
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10, fill=[128,128,128]))
        elif args.dataset == 'svhn':
            train_transform.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN, fill=[128,128,128]))
        elif args.dataset == 'imagenet':
            train_transform.append(transforms.RandAugment(num_ops=2, magnitude=20, fill=[128,128,128])) # [heavy2] in How to train your ViT?
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    if args.dataset == 'imagenet':
        test_transform += [
            transforms.Resize(int(args.size/0.875)),
            transforms.CenterCrop(args.size)
        ]
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    
_DOWNLOAD = True
_TORCHVISION_DATA_DIR = "./data" 
_IMAGENET_DATA_DIR = ""

def get_dataset(args):
    
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(_TORCHVISION_DATA_DIR, train=True, transform=train_transform, download=_DOWNLOAD)
        test_ds = torchvision.datasets.CIFAR10(_TORCHVISION_DATA_DIR, train=False, transform=test_transform, download=_DOWNLOAD)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(_TORCHVISION_DATA_DIR, train=True, transform=train_transform, download=_DOWNLOAD)
        test_ds = torchvision.datasets.CIFAR100(_TORCHVISION_DATA_DIR, train=False, transform=test_transform, download=_DOWNLOAD)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(_TORCHVISION_DATA_DIR, split="train",transform=train_transform, download=_DOWNLOAD)
        test_ds = torchvision.datasets.SVHN(_TORCHVISION_DATA_DIR, split="test", transform=test_transform, download=_DOWNLOAD)

    elif args.dataset == "imagenet":
        args.in_c = 3
        args.num_classes=1000
        args.size = 224
        args.padding = 0
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transform, test_transform = get_transform(args)

        train_ds = torchvision.datasets.ImageFolder(os.path.join(_IMAGENET_DATA_DIR, "train"), transform=train_transform)
        test_ds = torchvision.datasets.ImageFolder(os.path.join(_IMAGENET_DATA_DIR, "val"), transform=test_transform)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    experiment_name+=f"_p={args.patch}"
    experiment_name+=f"_nl={args.num_layers}"
    experiment_name+=f"_hd={args.head}"
    experiment_name+=f"_h={args.hidden}"
    experiment_name+=f"_mlp={args.mlp_hidden}"
    experiment_name+=f"_wd={args.weight_decay}"
    experiment_name+=f"_bs={args.batch_size}"
    experiment_name+=f"_e={args.max_epochs}"
    experiment_name+=f"_seed={args.seed}"
    print(f"Experiment:{experiment_name}")
    return experiment_name
