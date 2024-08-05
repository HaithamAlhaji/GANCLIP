import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:

    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import clip as clip

# Retrieves fixed batches of training and testing data.
def get_fix_data(train_dl, test_dl, text_encoder, args):
    """
    Retrieves and concatenates one batch of training and testing data, including images, sentence embeddings,
    word embeddings, and random noise.

    Parameters:
    train_dl (DataLoader): Dataloader for the training dataset.
    test_dl (DataLoader): Dataloader for the testing dataset.
    text_encoder (nn.Module): Model or function to encode text captions.
    args (Namespace): Arguments containing configuration settings, such as device and z_dim.

    Returns:
    tuple: Concatenated images, sentence embeddings, word embeddings, and random noise.
    """
    fixed_image_train, _, _, fixed_sent_train, fixed_word_train, fixed_key_train = get_one_batch_data(train_dl, text_encoder, args)
    fixed_image_test, _, _, fixed_sent_test, fixed_word_test, fixed_key_test= get_one_batch_data(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    return fixed_image, fixed_sent, fixed_word, fixed_noise

# Gets one batch of data from a dataloader.
def get_one_batch_data(dataloader, text_encoder, args):
    """
    Retrieves one batch of data from the dataloader and prepares it by encoding the text captions.

    Parameters:
    dataloader (DataLoader): Dataloader to retrieve the batch of data from.
    text_encoder (nn.Module): Model or function to encode text captions.
    args (Namespace): Arguments containing configuration settings, such as device.

    Returns:
    tuple: Batch of images, captions, CLIP tokens, sentence embeddings, word embeddings, and keys.
    """
    data = next(iter(dataloader))
    imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, args.device)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys

# Prepares data by encoding the text captions
def prepare_data(data, text_encoder, device):
    """
    Prepares the data by moving images and CLIP tokens to the specified device and encoding the text captions.

    Parameters:
    data (tuple): A batch of data containing images, captions, CLIP tokens, and keys.
    text_encoder (nn.Module): Model or function to encode text captions.
    device (torch.device): The device (CPU or GPU) to which the data should be moved.

    Returns:
    tuple: Images, captions, CLIP tokens, sentence embeddings, word embeddings, and keys.
    """
    imgs, captions, CLIP_tokens, keys = data
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys

# Encodes text captions using a text encoder (e.g., CLIP)
def encode_tokens(text_encoder, caption):
    """
    Encodes text captions using the provided text encoder to obtain sentence and word embeddings.

    Parameters:
    text_encoder (nn.Module): Model or function to encode text captions.
    caption (Tensor): The input text captions to be encoded.

    Returns:
    tuple: Sentence embeddings and word embeddings.
    """
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 

# Loads and processes images
def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    """
    Loads an image from the given path, optionally crops it using the bounding box, and applies transformations.

    Parameters:
    img_path (str): Path to the image file.
    bbox (list or None): Bounding box to crop the image [x, y, width, height]. If None, no cropping is done.
    transform (callable or None): Transformations to apply to the image. If None, no transformations are applied.
    normalize (callable or None): Normalization to apply to the image. If None, no normalization is applied.

    Returns:
    Image: Transformed (and possibly cropped and normalized) image.
    """
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img

# Reads and processes captions from a text file
def get_caption(cap_path,clip_info):
    """
    Reads captions from a file, selects one at random, and tokenizes it using the CLIP tokenizer.

    Parameters:
    cap_path (str): Path to the file containing captions.
    clip_info (object): CLIP model information for tokenization.

    Returns:
    tuple: Selected caption and its tokenized form.
    """
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    sent_ix = random.randint(0, len(eff_captions))
    caption = eff_captions[sent_ix]
    tokens = clip.tokenize(caption,truncate=True)
    return caption, tokens[0]


################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):

    # Initializes the dataset with parameters for transformation, data directory, and other configurations.
    def __init__(self, split, transform=None, args=None):
        self.transform = transform
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split=split
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.filenames = self.load_filenames(self.data_dir, split)
        self.number_example = len(self.filenames)

    # Loads bounding box information if available.
    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    # Loads filenames of images from a pickle file.
    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    # Retrieves an item from the dataset, including image and corresponding caption
    def __getitem__(self, index):
        #
        key = self.filenames[index]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        #
        if self.dataset_name.lower().find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/jpg/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/jpg/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('cc3m') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
        elif self.dataset_name.lower().find('cc12m') != -1:
            if self.split=='train':
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
        else: # birds CUB_200_2011
            img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
            text_name = '%s/text/%s.txt' % (data_dir, key)
        #
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        caps,tokens = get_caption(text_name,self.clip4text)
        return imgs, caps, tokens, key

    # Returns the number of examples in the dataset
    def __len__(self):
        return len(self.filenames)