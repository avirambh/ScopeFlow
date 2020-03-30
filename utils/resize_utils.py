import cv2
import torch
import numpy as np
from math import ceil


def resize_np(images, resize_shape=(436, 1024)):
    new_images = np.stack([cv2.resize(images[ix], resize_shape)
                           for ix in range(len(images))])
    return new_images


def resize_tensor(ten, size):
    return torch.nn.functional.interpolate(input=ten,
                                           size=size,
                                           mode='bilinear',
                                           align_corners=False)


def get_orig_image(im, size, to_numpy=False):
    imtmp = resize_tensor(im, size)
    im = torch.flip(imtmp, [1])
    if to_numpy:
        return im.detach().cpu().numpy()
    return im.cuda()


def resize_flow(prediction, orig_size, divisor=64., use_np=False):

    if prediction.shape[-1] == orig_size[-1] and prediction.shape[-2] == orig_size[-2]:
        return prediction

    H = orig_size[0]
    W = orig_size[1]

    # H_, W_ = prediction.shape[2:]
    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)

    if use_np:
        prediction = np.swapaxes(np.swapaxes(prediction, 1, 2), 2, 3)
        prediction = resize_np(prediction, resize_shape=(W, H)).transpose(0, 3, 1, 2)
    else:
        prediction = resize_tensor(prediction, (H, W))

    prediction[:, 0, :, :] *= W / float(W_)
    prediction[:, 1, :, :] *= H / float(H_)
    return prediction