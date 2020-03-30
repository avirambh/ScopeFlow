import sklearn as sk  # Solving static TLS
import torch
import numpy as np
from scipy import ndimage
from sklearn.metrics import f1_score
from utils import flowlib


def minimax(im, per_image=True):
    if per_image:
        batch_size = im.shape[0]
        min_vals = im.min(1)[0].min(1)[0].min(1)[0].reshape(batch_size, 1, 1, 1)
        max_vals = im.max(1)[0].max(1)[0].max(1)[0].reshape(batch_size, 1, 1, 1)
    else:
        min_vals = im.min()
        max_vals = im.max()
    return (im - min_vals) / (max_vals - min_vals)


def get_mepe(flow, prediction):
    if type(flow) is torch.Tensor:
        flow = flow.detach().cpu().numpy()

    if type(prediction) is torch.Tensor:
        prediction = prediction.detach().cpu().numpy()

    epes = []
    mepes = []
    for ix in range(len(prediction)):
        cmepe, cepe = flowlib.flow_error(flow[ix, 0, :, :],
                                         flow[ix, 1, :, :],
                                         prediction[ix, 0, :, :],
                                         prediction[ix, 1, :, :])
        mepes.append(cmepe)
        epes.append(cepe)
    return mepes, epes


def get_occ_mepe(flow, prediction, occ, dilate=0):
    """
    Calculate different mean EPEs

    :param flow: GT flow
    :param prediction: Predicted flow
    :param occ: occlusions map (3D)
    :param dilate: number of dilation iteration (default=0)
    :return: mepe, mepe_occ, mepe_no_occ
    """
    if type(occ) == torch.Tensor:
        occ = occ.detach().cpu().numpy()
        if len(occ.shape) == 4:
            occ = occ.squeeze(1)

    # Get MEPE
    mepes, epes = get_mepe(flow, prediction)

    # Create occlusions map
    if dilate > 0:
        occ = ndimage.morphology.binary_dilation(occ, iterations=dilate)
    occ_mask = (occ > 0)
    num_occ_pixels = [np.sum(occ_mask[ix]) for ix in range(len(occ_mask))]
    all_pixels = [cocc.shape[0] * cocc.shape[1] for cocc in occ]

    # Get MEPEs
    mepes_occ = []
    mepes_no_occ = []
    for ix, epe in enumerate(epes):
        if num_occ_pixels[ix]:
            mepe_occ = np.sum(np.multiply(epe, occ_mask[ix])) / float(num_occ_pixels[ix])
            mepes_occ.append(mepe_occ)
        mepe_no_occ = np.sum(np.multiply(epe, 1 - occ_mask[ix])) / \
                      float(all_pixels[ix] - num_occ_pixels[ix])

        mepes_no_occ.append(mepe_no_occ)

    return {'mepes': mepes, 'mepes_occ': mepes_occ, 'mepes_no_occ': mepes_no_occ}


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def get_occ_err(gt_occ, occ_prediction, aggregate=False, metric='f1'):
    if len(gt_occ.shape) == 3:
        gt_occ = torch.Tensor(gt_occ).unsqueeze(1).detach().cpu().numpy()
    if not type(occ_prediction) is np.ndarray:
        occ_prediction = occ_prediction.detach().cpu().numpy()

    if metric == 'iou':
        similar_labels = gt_occ == occ_prediction
        res = [(1 - np.sum(similar_labels[i]) / float(gt_occ[i].size))
               for i in range(len(gt_occ))]
    elif metric == 'f1':
        res = [f1_score(occ_prediction[i, 0], gt_occ[i, 0], average='micro')
               for i in range(len(gt_occ))]  # [0.44158321411540297, 0.6779326299159284]
    if aggregate:
        return np.mean(res)
    return res
