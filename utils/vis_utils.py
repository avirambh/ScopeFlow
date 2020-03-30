import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import flowlib
from utils.resize_utils import resize_tensor
from utils.image_utils import rgb2gray, warp_image, torchify
# from utils.torch_utils import forward_and_eval
plt.rcParams["axes.grid"] = False
EPS = 1e-5

################# Flow and images ##################
def get_batch_flow_images(flow_batch):
    res = np.stack([flowlib.flow_to_image(input.detach().cpu().numpy().transpose(1, 2, 0))
                    for input in flow_batch])
    return res


def show_velocity(flow, velocity):
    plt.subplot(2, 1, 1)
    plt.title('Flow')
    show_flow(flow, subplote=True)
    plt.subplot(2, 1, 2)
    plt.title('Velocity')
    show_image(velocity, subplote=True)
    plt.show()


def show_flow(flow, subplote=False, occ=None):
    if flow is None:
        print("Flow is None")
        return

    if occ is not None:
        occ = occ.transpose(1, 2, 0)

    # Take first image
    if len(flow.shape) == 4:
        flow = flow[0]

    # Move to numpy
    if type(flow) is torch.Tensor:
        flow = flow.detach().cpu().numpy()

    # Change to channel last
    if flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)

    flowlib.visualize_flow(flow,
                           subplote=subplote,
                           occ=occ)


def show_image(im, subplote=False, overlay=False, ax=True):
    if type(im) is not np.ndarray:
        im = im.detach().cpu().numpy()

    if overlay:
        alpha = 0.5
    else:
        alpha = 1

    if im.max() > 1.:
        im = im / 255.

    if len(im.shape) == 4:  # Print first
        im = im[0]

    if len(im.shape) == 3:
        if im.shape[0] == 1:  # Grayscale
            plt.imshow(im[0], alpha=alpha)
        else:  # RGB
            if im.shape[2] > 3:  # 3'rd channel is W
                im = im.transpose(1, 2, 0)
            plt.imshow(im, alpha=alpha)
    else:
        plt.imshow(im, alpha=alpha)
    if not ax:
        plt.axis('off')

    if not subplote:
        plt.show()


def plot_occ_error(occ_prediction, gt_occ):
    plt.subplot(121)
    plt.title('Pred')
    show_image(occ_prediction, subplote=True)
    plt.subplot(122)
    plt.title('GT')
    show_image(gt_occ)


def plot_inputs(data):
    plt.subplot(331)
    plt.title('input_im1')
    show_image(data['input1'], subplote=True)
    plt.subplot(332)
    plt.title('input_im2')
    show_image(data['input2'], subplote=True)
    plt.subplot(333)
    plt.title('input_flow_gt')
    show_flow(data['target1'], subplote=True)
    plt.subplot(334)
    plt.title('input_occ_gt')
    show_image(data['target_occ1'], subplote=True)
    if 'invalid' in data:
        plt.subplot(335)
        plt.title('input_invalid_pixels')
        show_image(data['invalid'], subplote=True)
    plt.subplot(336)
    plt.title('input_occ_orig')
    show_image(data['target_occ1_orig'], subplote=True)
    plt.subplot(337)
    plt.title('input_im1_orig')
    show_image(data['input1_orig'], subplote=True)
    plt.subplot(338)
    plt.title('input_im2_orig')
    show_image(data['input2_orig'], subplote=True)
    plt.subplot(339)
    plt.title('input_flow_orig')
    show_flow(data['target1_orig'], subplote=True)

    plt.subplots_adjust(hspace=0.5)
    mng = plt.get_current_fig_manager()
    mng.resize(1200, 800)
    plt.show()


def plot_occ_inpainting(im1, im2, occ, gt_flow, res, occ_th=0.0):
    occ = torchify(occ).unsqueeze(0)
    occ_est = torch.Tensor(np.array(res['occs'][0], dtype=np.uint8)).cuda()
    flow_orig = torchify(res['flows'][0])
    flow_inp = torchify(res['interp_flows'][0])
    diff = (flow_inp - flow_orig).norm(2, keepdim=True, dim=1)
    occ = occ
    if occ.shape[-1] <= 3:
        occ = occ.transpose(2, 3).transpose(1, 2)
    occ = resize_tensor(occ, occ_est.shape[2:])[:, 0, :, :].unsqueeze(1) > 0
    occ_diff = (occ.float() - occ_est.float())

    plt.subplot(331)
    plt.title('Image 1')
    show_image(im1, subplote=True)

    plt.subplot(332)
    plt.title('Image 2')
    show_image(im2, subplote=True)

    plt.subplot(333)
    plt.title('GT flow')
    show_flow(gt_flow, subplote=True)

    plt.subplot(334)
    plt.title('Flow before refinement')
    show_flow(flow_orig, subplote=True)

    plt.subplot(335)
    plt.title('Flow after refinement')
    show_flow(flow_inp, subplote=True)

    plt.subplot(336)
    plt.title('Inpainting change magnitude')
    show_image(diff, subplote=True)

    plt.subplot(337)
    plt.title('Occlusions')
    show_image(occ, subplote=True)

    plt.subplot(338)
    plt.title('Occlusions est (th={})'.format(occ_th))
    show_image(occ_est, subplote=True)

    plt.subplot(339)
    plt.title('Occlusions diff')
    show_image(occ_diff)


def show_heatmaps(im, flow, prediction, grad_im):
    """

    :param im:
    :param flow:
    :param prediction:
    :param grad_im:
    :return:
    """
    plt.subplot(2, 2, 1)
    plt.title('Image')
    show_image(im, subplote=True)

    plt.subplot(2, 2, 2)
    plt.title('GT Flow')
    show_flow(flow, subplote=True)

    plt.subplot(2, 2, 3)
    plt.title('Flow error')
    show_flow(flow - prediction, subplote=True)

    plt.subplot(2, 2, 4)
    plt.title('Gradient')
    plt.imshow(rgb2gray(grad_im))

    plt.show()


def plot_summary(im1, im2, occ, flow, prediction, prediction_back, mepe, show_occ=False):
    """

    :param im1:
    :param im2:
    :param occ:
    :param flow:
    :param prediction:
    :return:
    """
    plt.subplot(2, 3, 1)
    plt.title('First image')
    show_image(im1, subplote=True)

    plt.subplot(2, 3, 2)
    plt.title('Second image')
    show_image(im2, subplote=True)

    plt.subplot(2, 3, 3)
    occ_percent = occ.sum() / (occ.shape[1] * occ.shape[2])
    if occ_percent < 1.:
        occ_percent *= 100.
    plt.title('Occlusions ({:.3}% of the image)'.format(occ_percent))
    show_image(occ, subplote=True)

    plt.subplot(2, 3, 4)
    plt.title('True flow')
    show_flow(flow, subplote=True)

    plt.subplot(2, 3, 5)
    plt.title('Predicted flow (MEPE: {:.3} px)'.format(float(mepe)))
    show_flow(prediction, subplote=True)

    plt.subplot(2, 3, 6)
    flow_diff = flow - prediction
    normed = np.sqrt(flow_diff[0, 0].__pow__(2) + flow_diff[0, 1].__pow__(2))
    plt.title('Flow error norm')
    show_image(normed, subplote=True)

    plt.show()


def plot_warped_images(warp_func, im1, im2, tprediction, flow, occ):
    occ = occ.transpose(1, 2, 0)
    flow = torchify(flow)
    warped_pred = warp_image(warp_func, im2, tprediction, im2compare=im1,
                             show_image=False)
    warped_true = warp_image(warp_func, im2, flow, im2compare=im1,
                             show_image=False)

    plt.subplot(2, 3, 1)
    plt.title('Predicted warping')
    plt.imshow(warped_pred)

    plt.subplot(2, 3, 2)
    plt.title('Ground Truth warping')
    plt.imshow(warped_true)

    plt.subplot(2, 3, 3)
    plt.title('Ground Truth warping masked with non occluded')
    plt.imshow(warped_true * (1 - occ))

    plt.subplot(2, 3, 4)
    plt.title('Predicted flow')
    show_flow(tprediction, subplote=True)

    plt.subplot(2, 3, 5)
    plt.title('GT flow')
    show_flow(flow, subplote=True)

    plt.subplot(2, 3, 6)
    plt.title('Flow error norm')
    flow_diff = flow - tprediction
    normed = torch.sqrt(flow_diff[0, 0].__pow__(2) + flow_diff[0, 1].__pow__(2))
    show_image(normed, subplote=True)

    plt.show()


def plot_occlusions_summary(im1, im2, occ, flow, filled_flow, edges):
    plt.subplot(2, 3, 1).set_axis_off()
    plt.title('First image')
    show_image(im1, subplote=True)
    plt.subplot(2, 3, 2).set_axis_off()
    plt.title('Second image')
    show_image(im2, subplote=True)
    plt.subplot(2, 3, 3).set_axis_off()
    occ_percent = 100 * (occ.sum() / (occ.shape[1] * occ.shape[2]))
    plt.title('GT Occlusions ({:.3}% of the image)'.format(occ_percent))
    show_image(occ, subplote=True)
    plt.subplot(2, 3, 4).set_axis_off()
    plt.title('True flow')
    show_flow(flow, subplote=True)
    plt.subplot(2, 3, 5).set_axis_off()
    plt.title('Occlusions masked true flow')
    show_flow(flow, subplote=True, occ=occ)
    plt.subplot(2, 3, 6).set_axis_off()
    plt.title('Occlusions inpainted flow')
    show_flow(filled_flow, subplote=True)

    plt.show()


def plot_refinement_results(out_interponet, pwc_flow, pwc_occ):
    res = out_interponet[0]
    show_flow(pwc_flow)
    if pwc_occ is not None:
        show_flow(pwc_flow * (1 - (pwc_occ / 255.)))
    show_flow(res)


def plot_refinement_iterations(res, refinements, gt_flow, args, rows=2):

    if gt_flow is not None:
        num_plots = args.refinement_passes + 2
    else:
        num_plots = args.refinement_passes + 1

    cols = np.ceil(num_plots / rows)
    plt.subplot(rows, cols, 1)
    plt.title('Backbone_output')
    show_flow(res['flows'], subplote=True)

    for ix, r in enumerate(refinements):
        offset = 2
        pos = ((ix + offset) % cols) + cols * ((ix + offset) // cols)
        plt.subplot(rows, cols, pos)
        plt.title('Refinement number {}'.format(ix))
        show_flow(r, subplote=True)

    if gt_flow is not None:
        offset = 3
        pos = ((ix + offset) % cols) + cols * ((ix + offset) // cols)
        plt.subplot(rows, cols, pos)
        plt.title('GT Flow')
        show_flow(gt_flow, subplote=True)

    plt.show()


def get_heatmaps(im1, im2, prediction, flow, tflow):
    raise Exception("Get heatmap is deprecated")
    # im1.requires_grad = True
    # predicted_flow, dur = forward_and_eval(im1, im2)
    # loss = torch.norm(predicted_flow - tflow[0], p=2, dim=1).mean()
    # loss = (predicted_flow - tflow[0]).pow(2).mean()
    # loss.backward()
    # grad_im = im1.grad.detach().cpu().numpy()
    # show_heatmaps(im2, flow, prediction, grad_im)
    # im1.requires_grad = False


def stitch_images(inputs, outputs, img_per_row=1):
    gap = 5
    height, width = inputs[0][0, :, :, 0].shape
    images = [*inputs, *outputs]
    columns = len(images)
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1),
                            height * (int(len(inputs) / 2*img_per_row))))

    for ix in range(0, 1):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            cur_im = np.array(images[cat][ix])
            if cur_im.max() <= 1.:
                cur_im *= 255
            im = cur_im.astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img

################# GRAPHS ##################
def plot_disponorm(flow_norm, disp_norm, title_suffix='', xlabel='', ylabel='',
                   subplot=[1, 2, 1]):
    """

    :param flow_norm:
    :param disp_norm:
    :param title_suffix:
    :param xlabel:
    :param ylabel:
    :param subplot:
    :return:
    """

    # Flatten
    flow_norm_vector = flow_norm.flatten().astype(int)
    displacement_vector = disp_norm.flatten().astype(int)

    # Get mean displacement per flow
    unique_flows = np.array([i for i in range(0, np.amax(flow_norm_vector) + 1)])
    flows_count = np.histogram(flow_norm_vector, bins=np.array(
        [i for i in range(0, np.amax(flow_norm_vector) + 2)]))[0]

    # Plot
    binned_error = np.bincount(flow_norm_vector, weights=displacement_vector)
    mean_disp_per_flow = binned_error / (flows_count + EPS)
    plot_graph(x=unique_flows, y=binned_error,
               title=('Sum of all errors ' + title_suffix),
               xlabel=xlabel or 'GT speed norm (Pixel per frame)',
               ylabel='Sum of prediction error norms (Pixel per frame)',
               subplot=subplot)
    plot_graph(x=unique_flows, y=mean_disp_per_flow,
               title='Mean error ' + title_suffix,
               xlabel=xlabel or 'GT speed norm (Pixel per frame)',
               ylabel='Mean of prediction error norms (Pixel per frame)',
               subplot=[subplot[0], subplot[1], subplot[2] + 1])

    # normed_mean_disp_per_flow = mean_disp_per_flow / (unique_flows + EPS)
    # plot_graph(x=unique_flows, y=normed_mean_disp_per_flow,
    #            title='Normalized (by GT flow value) mean error ' + title_suffix,
    #            xlabel=xlabel or 'GT speed norm (Pixel per frame)',
    #            ylabel='Normalized (by GT flow value) mean of prediction error norms
    #            (Pixel per frame)')


def plot_graph(x, y, title, xlabel, ylabel, subplot=None, show=True):
    """

    :param x:
    :param y:
    :param title:
    :param xlabel:
    :param ylabel:
    :param subplot:
    :param show:
    :return:
    """
    if subplot is not None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show and subplot is None or (not subplot[0] * subplot[1] != subplot[2]):
        plt.show()


def plot_graphs(metrics, disponorm=False, dispoangle=False, dispolap=False,
                mepe_max_flow=False, mepe_occ_perc=False):
    if disponorm:
        # Plot regular
        plot_disponorm(metrics['flow_norms'], metrics['displacements_norm'],
                       title_suffix='per GT speed over all pixels')

        # Plot norm by occluded
        occ_mask = (metrics['occ'].flatten() > 0).astype(int)
        speed_no_occ = metrics['flow_norms'].flatten()[occ_mask > 0]
        disp_err_no_occ = metrics['displacements_norm'].flatten()[occ_mask > 0]
        plot_disponorm(speed_no_occ, disp_err_no_occ,
                       title_suffix='per GT speed over all pixels (without occluded pixels)')

    if dispoangle:
        # Plot regular
        plot_disponorm(metrics['flow_angles'], metrics['displacements_norm'],
                       title_suffix='per GT angle over all pixels',
                       xlabel='GT flow angle', ylabel='Predicted speed error (norm)')

        # Plot norm by occluded
        occ_mask = (metrics['occ'].flatten() > 0).astype(int)
        speed_no_occ = metrics['flow_angles'].flatten()[occ_mask > 0]
        disp_err_no_occ = metrics['displacements_norm'].flatten()[occ_mask > 0]
        plot_disponorm(speed_no_occ, disp_err_no_occ,
                       title_suffix='per GT angle over all pixels (without occluded pixels)',
                       xlabel='GT flow angle', ylabel='Predicted speed error (norm)')

    if dispolap:
        pass
