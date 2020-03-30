import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import flowlib
from models.pwc_modules import WarpingLayer
from utils.flowlib import save_flow_image


def to_image_dims(img, use_np=True, aug_trans=True):
    # Verify torch
    if type(img) != torch.Tensor:
        img = torch.Tensor(img)

    # Move to 3 dims
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    elif len(img.shape) == 4 and img.shape[0] == 1:
        img = img[0]

    # Prepare to image libraries for augmentation
    if img.shape[0] <= 3 and aug_trans:
        img = img.transpose(0, 1).transpose(1, 2)
    elif img.shape[-1] <= 3 and not aug_trans:
        img = img.transpose(1, 2).transpose(0, 1)

    # Convert to ndarray
    if use_np:
        img = img.detach().cpu().numpy()

    return img


def verify_dims(data, use_np=True, aug_trans=True):
    for k, v in data.items():
        if type(v) == torch.Tensor or type(v) == np.ndarray:
            data[k] = to_image_dims(v, use_np=use_np, aug_trans=aug_trans)
    return data


def rgb2gray(rgb):
    """

    :param rgb:
    :return:
    """

    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def warp_np_images(im, flow):
    im1w = WarpingLayer().cuda()(torch.Tensor(im).cuda().unsqueeze(0),
                                 torch.Tensor(flow).unsqueeze(0).cuda())
    return im1w.detach().cpu().numpy()[0]


def torchify(arr):
    if not type(arr) == torch.Tensor or not arr.is_cuda:
        return torch.Tensor(arr).cuda()
    return arr


def warp_image(warp_func, im2warp_back, flow, im2compare=None, show_image=True):
    if len(flow.shape) == 3:
        flow = torch.Tensor(flow).unsqueeze(0)
    im2warp_back = torchify(im2warp_back)
    flow = torchify(flow)
    warped = warp_func(im2warp_back, flow)

    if im2compare is not None:
        im2show = warped.detach().cpu() * 0.5 + im2compare[0] * 0.5
        warped = im2show[0].numpy().transpose(1, 2, 0) / 255.0
        if show_image:
            plt.imshow(warped)
        return warped
    return warped


def my_make_dir(dname):
    if not os.path.isdir(dname):
        print("Creating directory: {}".format(dname))
        os.makedirs(dname)


def save_and_load(current_flow_name, prediction_tmp, out_dir, debug=True):
    full_flow_path = os.path.join(out_dir, current_flow_name)
    my_make_dir(os.path.dirname(full_flow_path))

    assert len(prediction_tmp) == 1
    orig_batch_size = len(prediction_tmp.shape)
    if orig_batch_size > 3:
        prediction_tmp = prediction_tmp[0]
    if debug:
        print("Writing {}".format(full_flow_path))
    flowlib.write_flow(prediction_tmp.transpose(1, 2, 0), full_flow_path)
    if debug:
        print("Reading {}".format(full_flow_path))
    prediction = flowlib.read_flo_file(full_flow_path).transpose(2, 0, 1)
    if orig_batch_size > 3:
        prediction = np.expand_dims(prediction, 0)
    return prediction


def save_flow_viz(current_flow_name, prediction_tmp, out_dir, debug=True):
    dname = os.path.dirname(out_dir)  # output/exp_name, same occ for clean and final
    bname = os.path.basename(out_dir)  # output/exp_name, same occ for clean and final
    full_flow_path = os.path.join(dname, 'flow_viz', bname,
                                    current_flow_name.replace('.flo', '.png'))
    my_make_dir(os.path.dirname(full_flow_path))

    # Write and load
    assert len(prediction_tmp) == 1
    orig_batch_size = len(prediction_tmp.shape)
    if orig_batch_size > 3:
        prediction_tmp = prediction_tmp[0]

    if debug:
        print("Writing {}".format(full_flow_path))
    save_flow_image(prediction_tmp.transpose(1, 2, 0), full_flow_path)


def save_and_load_occ(current_flow_name, occ_est, out_dir, debug=False):
    dname = os.path.dirname(out_dir)  # output/exp_name, same occ for clean and final
    current_occ_name = os.path.join(dname, 'occlusions',
                                    current_flow_name.replace('.flo', '.png'))
    my_make_dir(os.path.dirname(current_occ_name))

    # Write and load
    assert len(occ_est) == 1
    assert len(occ_est.shape) == 4
    if debug:
        print("Writing {}".format(current_occ_name))
    imageio.imwrite(current_occ_name, occ_est[0][0] * 255)
    if debug:
        print("Reading {}".format(current_occ_name))
    loaded = imageio.imread(current_occ_name) / 255
    loaded = np.expand_dims(np.expand_dims(loaded, 0), 0)
    return np.array(loaded).astype(np.uint8)

def occ_to_mask(occ, return_np=False):
    tens_out = nn.Sigmoid()(occ)
    if return_np:
        return np.round(tens_out.expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
                [0, 2, 3, 1])) * 255
    return tens_out.round()