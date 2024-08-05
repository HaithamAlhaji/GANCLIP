import os, sys
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p, get_rank
from lib.datasets import prepare_data

from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist


############   GAN   ############
def train(dataloader, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, scaler_G, scaler_D, args):
    """
    Trains the GAN model for one epoch.

    Args:
        dataloader (DataLoader): PyTorch DataLoader providing the training data.
        netG (nn.Module): The generator network.
        netD (nn.Module): The discriminator network.
        netC (nn.Module): The auxiliary classifier network.
        text_encoder (nn.Module): The text encoder network.
        image_encoder (nn.Module): The image encoder network.
        optimizerG (optim.Optimizer): The optimizer for the generator.
        optimizerD (optim.Optimizer): The optimizer for the discriminator.
        scaler_G (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training (generator).
        scaler_D (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training (discriminator).
        args (argparse.Namespace): Command line arguments and hyperparameters.

    Returns:
        None. The function trains the GAN for one epoch and updates the model parameters in place.
    """
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, image_encoder = netG.train(), netD.train(), netC.train(), image_encoder.train()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        ##############
        # Train D  
        ##############
        optimizerD.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            # prepare_data
            real, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            real = real.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            words_embs = words_embs.requires_grad_()
            # predict real
            CLIP_real,real_emb = image_encoder(real)
            real_feats = netD(CLIP_real)
            pred_real, errD_real = predict_loss(netC, real_feats, sent_emb, negtive=False)
            # predict mismatch
            mis_sent_emb = torch.cat((sent_emb[1:], sent_emb[0:1]), dim=0).detach()
            _, errD_mis = predict_loss(netC, real_feats, mis_sent_emb, negtive=True)
            # synthesize fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = netG(noise, sent_emb)
            CLIP_fake, fake_emb = image_encoder(fake)
            fake_feats = netD(CLIP_fake.detach())
            _, errD_fake = predict_loss(netC, fake_feats, sent_emb, negtive=True)
        # MA-GP
        if args.mixed_precision:
            errD_MAGP = MA_GP_MP(CLIP_real, sent_emb, pred_real, scaler_D)
        else:
            errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real)
        # whole D loss
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
        # update D
        if args.mixed_precision:
            scaler_D.scale(errD).backward()
            scaler_D.step(optimizerD)
            scaler_D.update()
            if scaler_D.get_scale()<args.scaler_min:
                scaler_D.update(16384.0)
        else:
            errD.backward()
            optimizerD.step()
        ##############
        # Train G  
        ##############
        optimizerG.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            fake_feats = netD(CLIP_fake)
            output = netC(fake_feats, sent_emb)
            text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
            errG = -output.mean() - args.sim_w*text_img_sim
        if args.mixed_precision:
            scaler_G.scale(errG).backward()
            scaler_G.step(optimizerG)
            scaler_G.update()
            if scaler_G.get_scale()<args.scaler_min:
                scaler_G.update(16384.0)
        else:
            errG.backward()
            optimizerG.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Train Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()


def test(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    """
    Evaluate the performance of the GAN using FID and CLIP similarity metrics.

    Arguments:
    dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    text_encoder (nn.Module): Model to encode text data.
    netG (nn.Module): Generator network.
    PTM (nn.Module): Pre-trained model for evaluation (e.g., CLIP).
    device (torch.device): Device to run the computations on (e.g., 'cuda' or 'cpu').
    m1 (np.ndarray): Mean vector of real images' features for FID calculation.
    s1 (np.ndarray): Covariance matrix of real images' features for FID calculation.
    epoch (int): Current epoch number.
    max_epoch (int): Maximum number of epochs.
    times (int): Number of times to iterate over the dataset for evaluation.
    z_dim (int): Dimensionality of the noise vector for the generator.
    batch_size (int): Number of samples per batch.

    Returns:
    tuple:
        - FID (float): Frechet Inception Distance score between generated and real images.
        - TI_sim (float): Average cosine similarity between generated images and their corresponding text embeddings.
    """
    FID, TI_sim = calculate_FID_CLIP_sim(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size)
    return FID, TI_sim


def save_model(netG, netD, netC, optG, optD, epoch, multi_gpus, step, save_path):
    """
    Save the state of the models and optimizers to a file.

    Arguments:
    netG (nn.Module): Generator network.
    netD (nn.Module): Discriminator network.
    netC (nn.Module): Classifier network.
    optG (torch.optim.Optimizer): Optimizer for the generator.
    optD (torch.optim.Optimizer): Optimizer for the discriminator.
    epoch (int): Current epoch number.
    multi_gpus (bool): Flag indicating if multiple GPUs are being used.
    step (int): Current training step.
    save_path (str): Path to save the model state.

    Returns:
    None
    """
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_MP(img, sent, out, scaler):
    """
    Compute the Mixed-Precision Mode Magnitude-Adjusted Gradient Penalty (MA-GP).

    Arguments:
    img (torch.Tensor): The input images.
    sent (torch.Tensor): The sentence embeddings corresponding to the images.
    out (torch.Tensor): The output predictions from the discriminator.
    scaler (torch.cuda.amp.GradScaler): The gradient scaler used for mixed-precision training.

    Returns:
    torch.Tensor: The computed gradient penalty.
    """
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)                        
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def MA_GP_FP32(img, sent, out):
    """
    Compute the Full-Precision Mode Magnitude-Adjusted Gradient Penalty (MA-GP).

    Arguments:
    img (torch.Tensor): The input images.
    sent (torch.Tensor): The sentence embeddings corresponding to the images.
    out (torch.Tensor): The output predictions from the discriminator.

    Returns:
    torch.Tensor: The computed gradient penalty.
    """
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp):
    """
    Generate and save samples from the generator network.

    Arguments:
    dataloader (torch.utils.data.DataLoader): DataLoader for fetching batches of real images and text data.
    netG (torch.nn.Module): The generator network used to generate images.
    text_encoder (torch.nn.Module): The text encoder used to process text inputs.
    save_dir (str): Directory path where the generated images and captions will be saved.
    device (torch.device): Device to which the tensors will be moved (e.g., 'cuda' or 'cpu').
    multi_gpus (bool): Flag indicating if multi-GPU setup is used.
    z_dim (int): Dimensionality of the noise vector used for generating images.
    stamp (int): A timestamp or identifier for organizing output files.

    Returns:
    None: The function saves generated images and captions to disk and does not return any value.
    """
    netG.eval()
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        real, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            fake_imgs = torch.clamp(fake_imgs, -1., 1.)
            if multi_gpus==True:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            else:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            mkdir_p(batch_img_save_dir)
            vutils.save_image(fake_imgs.data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                ######################################################
                # (3) Save fake images
                ######################################################      
                if multi_gpus==True:
                    single_img_name = 'batch_%04d.png'%(j)
                    single_img_save_dir  = osp.join(save_dir, 'single', str('gpu%d'%(get_rank())), 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)
                else:
                    single_img_name = 'step_%04d.png'%(step)
                    single_img_save_dir  = osp.join(save_dir, 'single', 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)   
                mkdir_p(single_img_save_dir)   
                im.save(single_img_save_name)
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Step: %d' % (step))


def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    """
    Calculate the FID (Frechet Inception Distance) and CLIP similarity scores between real and fake images.

    Arguments:
    real_images (torch.Tensor): Tensor of real images with shape (N, C, H, W), where N is the number of images, 
                                C is the number of channels (3 for RGB), H is the height, and W is the width.
    fake_images (torch.Tensor): Tensor of fake images with shape (N, C, H, W), where N is the number of images, 
                                C is the number of channels (3 for RGB), H is the height, and W is the width.
    text_embeddings (torch.Tensor): Tensor of text embeddings with shape (N, D), where N is the number of images and 
                                     D is the dimensionality of the text embeddings.
    clip_model (torch.nn.Module): The CLIP model used to compute the similarity between image and text embeddings.
    device (torch.device): Device to which the tensors will be moved (e.g., 'cuda' or 'cpu').

    Returns:
    tuple: A tuple containing:
        - fid_score (float): The FID score between real and fake images, which measures the distance between
                             the distributions of the real and fake images in feature space.
        - clip_sim (float): The CLIP similarity score between real and fake images, which measures how well
                            the generated images align with the text descriptions.
    """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = 1 # dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb,eval=True).float()
                # norm_ip(fake_imgs, -1, 1)
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                if epoch==-1:
                    loop.set_description('Evaluating]')
                else:
                    loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    # CLIP-score
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    # dist.all_gather(CLIP_score_gather, clip_cos)
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value,clip_score


def calc_clip_sim(clip, fake, caps_clip, device):
    """
    Calculate the cosine similarity between the embeddings of fake images and text descriptions using a CLIP model.

    Arguments:
    clip_model (torch.nn.Module): The CLIP model used to encode images and text into embeddings.
    fake_images (torch.Tensor): Tensor of fake images with shape (N, C, H, W), where N is the number of images, 
                                C is the number of channels (3 for RGB), H is the height, and W is the width.
    text_embeddings (torch.Tensor): Tensor of text embeddings with shape (N, D), where N is the number of images 
                                     and D is the dimensionality of the text embeddings.
    device (torch.device): Device to which the tensors will be moved (e.g., 'cuda' or 'cpu').

    Returns:
    torch.Tensor: A tensor of cosine similarities with shape (N,), where N is the number of images. Each value 
                  represents the average cosine similarity between the embeddings of a fake image and its 
                  corresponding text description.
    """
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim


def sample_one_batch(noise, sent, netG, multi_gpus, epoch, img_save_dir, writer):
    """
    Generate and save a batch of images from a given batch of noise and text captions.

    Arguments:
    noise (torch.Tensor): Tensor of shape (N, z_dim) representing the noise vectors used to generate images, 
                          where N is the number of samples and z_dim is the dimensionality of the noise vector.
    captions (torch.Tensor): Tensor of shape (N, D) representing text embeddings for each sample, where D is 
                              the dimensionality of the text embeddings.
    netG (torch.nn.Module): The generator model used to produce fake images from noise and text embeddings.
    multi_gpus (bool): Flag indicating whether multiple GPUs are being used. If True, handles saving 
                       images for different GPUs separately.
    epoch (int): The current training epoch. Used to name the saved image file.
    img_save_dir (str): Directory path where the generated images will be saved.
    writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer instance for logging metrics (not used in this function).

    Returns:
    None
    """
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        netG.eval()
        with torch.no_grad():
            B = noise.size(0)
            fixed_results_train = generate_samples(noise[:B//2], sent[:B//2], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results_test = generate_samples(noise[B//2:], sent[B//2:], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)


def generate_samples(noise, caption, model):
    """
    Generate images from noise vectors and text embeddings using a given model.

    Arguments:
    noise (torch.Tensor): Tensor of shape (N, z_dim) where N is the number of samples and z_dim is the dimensionality 
                          of the noise vector. Each row represents a noise vector used as input to the generator.
    captions (torch.Tensor): Tensor of shape (N, D) representing text embeddings for each sample, where D is the 
                              dimensionality of the text embeddings. These embeddings guide the generation of images.
    model (torch.nn.Module): The generator model used to produce images. It takes noise and text embeddings as input 
                              and generates corresponding images.

    Returns:
    torch.Tensor: A tensor of shape (N, C, H, W) containing the generated images, where N is the number of samples, 
                  C is the number of color channels (e.g., 3 for RGB), H is the height of the images, and W is the 
                  width of the images.
    """
    with torch.no_grad():
        fake = model(noise, caption, eval=True)
    return fake


def predict_loss(predictor, img_feature, text_feature, negtive):
    """
    Compute the prediction and loss using a predictor model.

    Arguments:
    predictor (torch.nn.Module): The model used to predict the similarity or score between image features and text features.
    img_feature (torch.Tensor): Tensor of image features extracted by the image encoder. It has shape (N, F), where N is 
                                the number of samples and F is the feature dimensionality.
    text_feature (torch.Tensor): Tensor of text features representing the text embeddings. It has shape (N, F) where N 
                                  is the number of samples and F is the feature dimensionality.
    negtive (bool): A boolean flag indicating whether to compute the loss for negative cases (i.e., when the features 
                    do not match) or positive cases (i.e., when the features are expected to match).

    Returns:
    output (torch.Tensor): Tensor of shape (N, 1) containing the raw prediction scores from the predictor model for each 
                           sample, where N is the number of samples.
    err (torch.Tensor): Scalar tensor containing the computed loss value based on the prediction and whether the case is 
                        positive or negative. The loss is computed using hinge loss if negtive is True, or using 
                        the opposite hinge loss if negtive is False.
    """
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    """
    Compute the hinge loss between the predictions and the target labels.

    Arguments:
    output (torch.Tensor): Tensor of shape (N, 1) containing the raw prediction scores from the model, where N is 
                           the number of samples.
    negtive (bool): A boolean flag indicating whether to compute the loss for negative cases (i.e., when the features 
                    do not match) or positive cases (i.e., when the features are expected to match).

    Returns:
    torch.Tensor: Scalar tensor containing the computed hinge loss. The loss is computed as follows:
        - For positive cases (negtive=False), the hinge loss is computed as `mean(max(0, 1 - output))`.
        - For negative cases (negtive=True), the hinge loss is computed as `mean(max(0, 1 + output))`.
    """
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err


def logit_loss(output, negtive):
    """
    Compute the binary cross-entropy (BCE) loss between the predicted probabilities and the target labels.

    Arguments:
    output (torch.Tensor): Tensor of shape (N, 1) containing the raw prediction scores from the model, where N is 
                           the number of samples.
    negtive (bool): A boolean flag indicating whether to compute the loss for positive cases (i.e., where the features 
                    are expected to match) or negative cases (i.e., where the features do not match).

    Returns:
    torch.Tensor: Scalar tensor containing the computed binary cross-entropy loss. The loss is computed as follows:
        - For positive cases (negtive=False), the loss is computed as `BCE(output, real_labels)`.
        - For negative cases (negtive=True), the loss is computed as `BCE(output, fake_labels)`.
    """
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Frechet Inception Distance (FID) between two distributions.

    The FID measures the similarity between two distributions of images: one from the generated images and one from the real images.
    It is used to evaluate the quality of generated images by comparing the feature distributions extracted from the Inception network.

    Arguments:
    mu1 (np.ndarray): Mean vector of the feature distribution for the real images.
    sigma1 (np.ndarray): Covariance matrix of the feature distribution for the real images.
    mu2 (np.ndarray): Mean vector of the feature distribution for the generated images.
    sigma2 (np.ndarray): Covariance matrix of the feature distribution for the generated images.
    eps (float, optional): A small value to avoid numerical instability when computing the square root of the covariance matrix product. Default is 1e-6.

    Returns:
    float: The Frechet Inception Distance between the two distributions. A lower FID indicates better quality of generated images.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
