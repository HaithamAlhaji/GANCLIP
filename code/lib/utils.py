import os
import sys
import errno
import numpy as np
import numpy.random as random
import torch
from torch import distributed as dist
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import pprint
import datetime
import dateutil.tz
from PIL import Image

import importlib
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# Dynamically imports and returns model components from a specified module.
def choose_model(model):
    '''choose models
    '''
    model = importlib.import_module(".%s"%(model), "models")
    NetG, NetD, NetC, CLIP_IMG_ENCODER, CLIP_TXT_ENCODER = model.NetG, model.NetD, model.NetC, model.CLIP_IMG_ENCODER, model.CLIP_TXT_ENCODER
    return NetG,NetD,NetC,CLIP_IMG_ENCODER, CLIP_TXT_ENCODER

# Computes the number of parameters in a given model by summing the number of elements in each parameter tensor.
def params_count(model):
    model_size = np.sum([p.numel() for p in model.parameters()]).item()
    return model_size

# Generates a timestamp string in the format YYYY_MM_DD_HH_MM_SS.
def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp

# Creates a directory at the specified path, handling the case where the directory already exists.
def mkdir_p(path):
    try:
        # check if path exists
        # if not os.path.exists(path):
        #     os.makedirs(path)
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Loads mean and standard deviation values from a .npz file.
def load_npz(path):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s

# config
# Loads configuration from a YAML file and returns it as an EasyDict.
def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg

# Converts string representations of boolean values in a dictionary to actual boolean values.
def str2bool_dict(dict):
    for key,value in dict.items():
        if type(value)==str:
            if value.lower() in ('yes','true'):
                dict[key] = True
            elif value.lower() in ('no','false'):
                dict[key] = False
            else:
                None
    return dict

# Merges command-line arguments with settings from a YAML configuration file.
def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        # args.update(opt)
        args = str2bool_dict(args)
        args = edict(args)
    return args

# Saves arguments to a YAML file at the specified path.
def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()

# Reads lines from a text file and returns them as a list.
def read_txt_file(txt_file):
    content = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            content.append(line)
    return content

# Retrieves the rank of the current process in a distributed training setup. Returns 0 if distributed training is not available or not initialized.
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

# Loads state dictionaries into an optimizer.
def load_opt_weights(optimizer, weights):
    optimizer.load_state_dict(weights)
    return optimizer

# Loads model and optimizer state dictionaries from a checkpoint file.
def load_models_opt(netG, netD, netC, optim_G, optim_D, path, multi_gpus):
    checkpoint = torch.load(path, map_location=torch.device('cuda:0'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus)
    netD = load_model_weights(netD, checkpoint['model']['netD'], multi_gpus)
    netC = load_model_weights(netC, checkpoint['model']['netC'], multi_gpus)
    optim_G = load_opt_weights(optim_G, checkpoint['optimizers']['optimizer_G'])
    optim_D = load_opt_weights(optim_D, checkpoint['optimizers']['optimizer_D'])
    return netG, netD, netC, optim_G, optim_D

# Loads model state dictionaries from a checkpoint file.
def load_models(netG, netD, netC, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'])
    netD = load_model_weights(netD, checkpoint['model']['netD'])
    netC = load_model_weights(netC, checkpoint['model']['netC'])
    return netG, netD, netC

# Loads weights specifically for the generator model (netG) from a checkpoint.
def load_netG(netG, path, multi_gpus, train):
    checkpoint = torch.load(path, map_location="cpu")
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus, train)
    return netG

# Loads model weights, adjusting for multi-GPU training if necessary.
def load_model_weights(model, weights, multi_gpus, train=True):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model

# Saves model and optimizer states to a checkpoint file, including the epoch number.
def save_models_opt(netG, netD, netC, optG, optD, epoch, multi_gpus, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

# Saves model states to a checkpoint file.
def save_models(netG, netD, netC, epoch, multi_gpus, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}}
        torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

# Saves model states, optimizer states, and scaler states to a checkpoint file.
def save_checkpoints(netG, netD, netC, optG, optD, scaler_G, scaler_D, epoch, multi_gpus, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                "scalers": {"scaler_G": scaler_G.state_dict(), "scaler_D": scaler_D.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

# Writes contents to a text file.
def write_to_txt(filename, contents): 
    fh = open(filename, 'w') 
    fh.write(contents) 
    fh.close()


def read_txt_file(txt_file):
    # text_file: file path
    content = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            content.append(line)
    return content

# Saves an image tensor to a file after transforming it from the [-1, 1] range to [0, 255].
def save_img(img, path):
    im = img.data.cpu().numpy()
    # [-1, 1] --> [0, 255]
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)
    im.save(path)

# Preprocesses input tensors for use with the CLIP model. This involves resizing and normalizing the inputs.
def transf_to_CLIP_input(inputs):
    device = inputs.device
    if len(inputs.size()) != 4:
        raise ValueError('Expect the (B, C, X, Y) tensor.')
    else:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
        inputs = ((inputs+1)*0.5-mean)/var
        return inputs.float()


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
