from skimage.io import imread
import numpy as np
import os
from config import *

def batch_data(batch_sz, batch_imgs, masks):
    imgs = []
    msks = []

    for img_name in batch_imgs:
        img = imread(os.path.join(TRAIN_PATH, img_name))
        msk = concat_mask(masks[masks['ImageId'] == img_name]['EncodedPixels'])

        img = img[::IMG_SCALING[0], ::IMG_SCALING[1]]
        msk = msk[::IMG_SCALING[0], ::IMG_SCALING[1]]

        imgs.append(img)
        msks.append(msk)

        if len(imgs) >= batch_sz:
            yield np.stack(imgs, 0) / 255.0, np.stack(msks, 0).astype(np.float32)
            imgs, msks = [], []


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def concat_mask(mask_list):
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
