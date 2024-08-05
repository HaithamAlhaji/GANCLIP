import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip
import importlib
from lib.utils import choose_model


###########   preparation   ############
def load_clip(clip_info, device):
    """
    Loads a CLIP model onto the specified device.

    Args:
        clip_info (dict): Dictionary containing information about the CLIP model.
                          Expected key: 'type' which indicates the type of CLIP model to load.
        device (torch.device): The device to load the CLIP model onto (e.g., 'cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The loaded CLIP model ready for inference or evaluation.
    
    Example:
        clip_info = {'type': 'ViT-B/32'}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_clip(clip_info, device)
    """
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


def prepare_models(args):
    """
    Prepares and initializes the models required for training, including CLIP models and GAN models.
    
    Args:
        args (argparse.Namespace): A namespace containing the arguments and configurations needed 
                                   for model preparation. Expected attributes:
            - device (torch.device): The device to load models onto.
            - local_rank (int): The local rank for distributed training.
            - multi_gpus (bool): Whether to use multiple GPUs.
            - clip4trn (dict): Information about the training CLIP model.
            - clip4evl (dict): Information about the evaluation CLIP model.
            - model (str): The model architecture to be used.
            - nf (int): Number of features.
            - z_dim (int): Dimension of the latent vector.
            - cond_dim (int): Dimension of the conditioning vector.
            - imsize (int): Size of the input images.
            - ch_size (int): Number of image channels.
            - mixed_precision (bool): Whether to use mixed precision training.
            - train (bool): Whether the models are being prepared for training.
    
    Returns:
        tuple: A tuple containing the initialized and prepared models:
            - CLIP4trn (torch.nn.Module): The CLIP model for training.
            - CLIP4evl (torch.nn.Module): The CLIP model for evaluation.
            - CLIP_img_enc (torch.nn.Module): The image encoder initialized with the training CLIP model.
            - CLIP_txt_enc (torch.nn.Module): The text encoder initialized with the training CLIP model.
            - netG (torch.nn.Module): The generator model.
            - netD (torch.nn.Module): The discriminator model.
            - netC (torch.nn.Module): The conditioning model.
    """
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER = choose_model(args.model)

    # image encoder
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()

    # text encoder
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()

    # GAN models
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size, args.mixed_precision, CLIP4trn).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision).to(device)
    netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)
    print("ccccccccccc")
    print(args.train)

    if (args.multi_gpus) and (args.train):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = torch.nn.parallel.DistributedDataParallel(netG, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netC = torch.nn.parallel.DistributedDataParallel(netC, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC


def prepare_dataset(args, split, transform):
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    """
    Prepares a dataset with the specified split (train/test) and image transformations.
    
    Args:
        args (argparse.Namespace): A namespace containing the arguments and configurations needed 
                                   for dataset preparation. Expected attributes:
            - imsize (int): Size of the input images.
            - ch_size (int): Number of image channels.
        split (str): The dataset split to use ('train' or 'test').
        transform (torchvision.transforms.Compose): The image transformations to apply.
    
    Returns:
        dataset (torch.utils.data.Dataset): The prepared dataset with the specified split and transformations.
    """
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    """
    Prepares data loaders for training and validation datasets.
    
    Args:
        args (argparse.Namespace): A namespace containing the arguments and configurations needed 
                                   for data loader preparation. Expected attributes:
            - batch_size (int): The batch size for the data loaders.
            - num_workers (int): The number of worker processes for data loading.
            - multi_gpus (bool): Whether to use multiple GPUs.
        transform (torchvision.transforms.Compose, optional): The image transformations to apply. 
                                                              If None, default transformations will be used.
    
    Returns:
        tuple: A tuple containing:
            - train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
            - valid_dataloader (torch.utils.data.DataLoader): The data loader for the validation dataset.
            - train_dataset (torch.utils.data.Dataset): The training dataset.
            - valid_dataset (torch.utils.data.Dataset): The validation dataset.
            - train_sampler (torch.utils.data.DistributedSampler or None): The sampler for the training dataset (if using multiple GPUs).
    """
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # train dataloader
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    
    # valid dataloader
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=valid_sampler)
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler