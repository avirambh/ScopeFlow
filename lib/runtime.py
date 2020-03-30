## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import os
import colorama
import logging
import pickle
import collections
import scipy.misc
import torch
import torch.nn as nn
import numpy as np

from utils import logger, tools
from utils.tools import MovingAverage
from utils.flow import flow_to_png, flow_to_png_middlebury
from utils.flow import write_flow, write_flow_png
from utils.vis_utils import show_image, show_flow, plot_summary, plot_inputs, \
    plot_occlusions_summary
from utils.metrics import get_occ_mepe, get_occ_err
from utils.image_utils import occ_to_mask
from datasets.kitti_combined import read_png_flow

# --------------------------------------------------------------------------------
# Exponential moving average smoothing factor for speed estimates
# Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].
# --------------------------------------------------------------------------------
TQDM_SMOOTHING = 0


# -------------------------------------------------------------------------------------------
# Magic progressbar for inputs of type 'iterable'
# -------------------------------------------------------------------------------------------
def create_progressbar(iterable,
                       desc="",
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False):
    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    dim = colorama.Style.DIM
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s " % (cyan, reset, bright, reset)  # description
    bar_format += "{percentage:3.0f}%"  # percentage
    bar_format += "%s|{bar}|%s " % (dim, reset)  # bar
    bar_format += " {n_fmt}/{total_fmt}  "  # i/n counter
    bar_format += "{elapsed}<{remaining}"  # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "  # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)  # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,  # Prefix for the progress bar
        "total": len(iterable),  # The number of expected iterations
        "leave": True,  # Leave progress bar when done
        "miniters": 1 if train else None,  # Minimum display update interval in iterations
        "unit": unit,  # String be used to define the unit of each iteration
        "initial": initial,  # The initial counter value.
        "dynamic_ncols": True,  # Allow window resizes
        "smoothing": TQDM_SMOOTHING,  # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,  # Specify a custom bar string formatting
        "position": offset,  # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close
    }

    return tools.tqdm_with_logging(**tqdm_args)


def tensor2float_dict(tensor_dict):
    return {key: tensor.item() if type(tensor) == torch.Tensor else tensor
            for key, tensor in tensor_dict.items()}


def format_moving_averages_as_progress_dict(moving_averages_dict={},
                                            moving_averages_postfix="avg"):
    progress_dict = collections.OrderedDict([
        (key + moving_averages_postfix, "%1.4f" % moving_averages_dict[key].mean())
        for key in sorted(moving_averages_dict.keys())
    ])
    return progress_dict


def format_learning_rate(lr):
    if np.isscalar(lr):
        return "{}".format(lr)
    else:
        return "{}".format(str(lr[0]) if len(lr) == 1 else lr)


def save_outputs(_args, example_dict, output_dict, sanity_check=False):
    # save occ
    save_root_img = _args.save + '/img/'
    save_root_flo = _args.save + '/flo/'

    if _args.save_result_bidirection:
        flow_f = output_dict["flow"].data.cpu().numpy()
        flow_b = output_dict["flow_b"].data.cpu().numpy()
        b_size = output_dict["flow"].data.size(0)
    else:
        flow_f = output_dict["flow"].data.cpu().numpy()
        b_size = output_dict["flow"].data.size(0)

    if _args.save_result_occ:
        if _args.save_result_bidirection:
            output_occ = np.round(
                nn.Sigmoid()(output_dict["occ"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
                    [0, 2, 3, 1])) * 255
            output_occ_b = np.round(
                nn.Sigmoid()(output_dict["occ_b"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
                    [0, 2, 3, 1])) * 255
        else:
            output_occ = occ_to_mask(output_dict["occ"], return_np=True)

    # file names
    file_names_img = []
    file_names_flo = []
    for ii in range(0, b_size):
        if "basedir" in example_dict.keys():
            file_name_img = save_root_img + example_dict["basedir"][ii] + '/' + str(example_dict["basename"][ii])
            file_name_flo = save_root_flo + example_dict["basedir"][ii] + '/' + str(example_dict["basename"][ii])
            file_names_img.append(file_name_img)
            file_names_flo.append(file_name_flo)
        else:
            file_name_img = save_root_img + '/' + str(example_dict["basename"][ii])
            if 'full_basename' in example_dict:
                basename = example_dict["full_basename"][ii]
            else:
                basename = example_dict["basename"][ii]
            file_name_flo = save_root_flo + '/' + str(basename)
            file_names_img.append(file_name_img)
            file_names_flo.append(file_name_flo)

        directory_img = os.path.dirname(file_name_img)
        if not os.path.exists(directory_img):
            os.makedirs(directory_img)
        directory_flo = os.path.dirname(file_name_flo)
        if not os.path.exists(directory_flo):
            os.makedirs(directory_flo)

    if _args.save_result_img:
        for ii in range(0, b_size):
            if _args.save_result_occ:
                file_name_occ = file_names_img[ii] + '_occ.png'
                scipy.misc.imsave(file_name_occ, output_occ[ii])

                if _args.save_result_bidirection:
                    scipy.misc.imsave(file_names_img[ii] + '_occ_b.png', output_occ_b[ii])

            # Img vis
            if hasattr(_args, "save_input") and _args.save_input:
                show_image(example_dict['input1'][0].detach().cpu().numpy().transpose([1, 2, 0]))
                file_name_im1 = file_names_img[ii] + '_im1.png'
                file_name_im2 = file_names_img[ii] + '_im2.png'
                scipy.misc.imsave(file_name_im1,
                                  example_dict['input1'][0].detach().cpu().numpy().transpose([1, 2, 0]))
                scipy.misc.imsave(file_name_im2,
                                  example_dict['input2'][0].detach().cpu().numpy().transpose([1, 2, 0]))

            # flow vis
            flow_f_rgb = flow_to_png_middlebury(flow_f[ii, ...])
            file_name_flo_vis = file_names_img[ii] + '_flow.png'
            scipy.misc.imsave(file_name_flo_vis, flow_f_rgb)

            if _args.save_result_bidirection:
                flow_b_rgb = flow_to_png_middlebury(flow_b[ii, ...])
                file_name_flo_vis = file_names_img[ii] + '_flow_b.png'
                scipy.misc.imsave(file_name_flo_vis, flow_b_rgb)

    if _args.save_result_flo or _args.save_result_png:
        for ii in range(0, b_size):
            if _args.save_result_flo:
                file_name = file_names_flo[ii] + '.flo'
                write_flow(file_name, flow_f[ii, ...].swapaxes(0, 1).swapaxes(1, 2))
            if _args.save_result_png:
                file_name = os.path.basename(file_names_flo[ii]) + '.png'
                dir_name = os.path.dirname(file_names_flo[ii])
                if 'kitti_version' in example_dict:
                    dir_name = os.path.join(dir_name, example_dict['kitti_version'][0])
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                file_name = os.path.join(dir_name, file_name)
                cur_flow = flow_f[ii, ...].swapaxes(0, 1).swapaxes(1, 2)
                write_flow_png(file_name, cur_flow)

                # Sanity check
                if sanity_check:
                    print('Filename {}'.format(file_names_flo[ii]))
                    flo_read = read_png_flow(file_name)[0]
                    print(flo_read.shape)
                    diff = (flo_read - cur_flow)
                    print("Diff png is {}".format(diff.sum()))
                    # show_image(diff[:,:,0])


class TrainingEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 loader,
                 optimizer,
                 augmentation=None,
                 add_progress_stats={},
                 desc="Training Epoch"):

        self._args = args
        self._args.debug = hasattr(args, 'debug') and self._args.debug
        self._desc = desc
        self._loader = loader
        self._model_and_loss = model_and_loss
        self._optimizer = optimizer
        self._augmentation = augmentation
        self._add_progress_stats = add_progress_stats
        if 'parallel' in str(type(self._model_and_loss.model)):
            self.model = self._model_and_loss.model.module
        else:
            self.model = self._model_and_loss.model
        self.random_freeze = hasattr(self.model, 'random_freeze') \
                             and self.model.random_freeze
        self.freeze = self.random_freeze or (hasattr(args, 'freeze_list')
                                             and args.freeze_list)
        self._save_output = False
        if self._args.save_result_img or self._args.save_result_flo or self._args.save_result_png:
            self._save_output = True

    def _step(self, example_dict):

        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=False)

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self._augmentation is not None:
            with torch.no_grad():
                if self._args.debug:
                    example_dict.update({"{}_orig".format(k): v
                                         for k, v in example_dict.items()})
                example_dict = self._augmentation(example_dict)
                if self._args.debug:
                    plot_inputs(example_dict)

        # -------------------------------------------------------------
        # Convert inputs/targets to variables that require gradients
        # -------------------------------------------------------------
        for key, tensor in example_dict.items():
            if key in input_keys:
                example_dict[key] = tensor.requires_grad_(True)
            elif key in target_keys:
                example_dict[key] = tensor.requires_grad_(False)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        batch_size = example_dict["input1"].size()[0]

        # -------------------------------------------------------------
        # Reset gradients
        # -------------------------------------------------------------
        self._optimizer.zero_grad()

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        loss_dict, output_dict = self._model_and_loss(example_dict)

        # -------------------------------------------------------------
        # Optionally save output
        # -------------------------------------------------------------
        if self._save_output:
            vis_dict = {"flow": output_dict["flow"][-1][0],
                        "occ": output_dict["occ"][-1][0]
                        }
            save_outputs(self._args, example_dict, vis_dict)

        # -------------------------------------------------------------
        # Check total_loss for NaNs
        # -------------------------------------------------------------
        training_loss = loss_dict[self._args.training_key]
        assert (not np.isnan(training_loss.item())), "training_loss is NaN"

        # -------------------------------------------------------------
        # Back propagation
        # -------------------------------------------------------------
        training_loss.backward()
        self._optimizer.step()

        # -------------------------------------------------------------
        # Return success flag, loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, output_dict, batch_size

    def run(self, offset=0):

        # ---------------------------------------
        # Tell model that we want to train
        # ---------------------------------------
        self._model_and_loss.train()

        # -------------------------------------------------------------
        # Handle freeze
        # -------------------------------------------------------------
        if self.random_freeze:
            self.model.freeze_random_weights()
        if self.freeze and hasattr(self.model, 'submodules_summary'):
            self.model.submodules_summary()

        # ---------------------------------------
        # Keep track of moving averages
        # ---------------------------------------
        moving_averages_dict = None

        # ---------------------------------------
        # Progress bar arguments
        # ---------------------------------------
        progressbar_args = {
            "iterable": self._loader,
            "desc": self._desc,
            "train": True,
            "offset": offset,
            "logging_on_update": False,
            "logging_on_close": True,
            "postfix": True
        }

        # ---------------------------------------
        # Perform training steps
        # ---------------------------------------
        with create_progressbar(**progressbar_args) as progress:
            for example_dict in progress:
                # perform step
                loss_dict_per_step, output_dict, batch_size = self._step(example_dict)
                # convert
                loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                # --------------------------------------------------------
                # Possibly initialize moving averages
                # --------------------------------------------------------
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }

                # --------------------------------------------------------
                # Add moving averages
                # --------------------------------------------------------
                for key, loss in loss_dict_per_step.items():
                    moving_averages_dict[key].add_average(loss, addcount=batch_size)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="_ema")

                progress.set_postfix(progress_stats)

        # -------------------------------------------------------------
        # Return loss and output dictionary
        # -------------------------------------------------------------
        ema_loss_dict = {key: ma.mean() for key, ma in moving_averages_dict.items()}
        return ema_loss_dict


class EvaluationEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 loader,
                 augmentation=None,
                 add_progress_stats={},
                 desc="Evaluation Epoch"):
        self._args = args
        self._desc = desc
        self._loader = loader
        self._model_and_loss = model_and_loss
        self._add_progress_stats = add_progress_stats
        self._augmentation = augmentation
        self._save_output = False
        if self._args.save_result_img or self._args.save_result_flo or self._args.save_result_png:
            self._save_output = True
        self.samples_mepe = {}
        self._args.white_list = hasattr(self._args, 'white_list') and self._args.white_list

    def _step(self, example_dict):
        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=False)

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        batch_size = example_dict["input1"].size()[0]

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        loss_dict, output_dict = self._model_and_loss(example_dict)

        # -------------------------------------------------------------
        # Analyze end point errors per occ / no occ
        # -------------------------------------------------------------
        if 'target1' in example_dict and 'target_occ1' in example_dict:
            loss_dict.update(get_occ_mepe(output_dict['flow'],
                                          example_dict['target1'],
                                          example_dict['target_occ1']))

        # Save epe per sample
        if hasattr(self._args, 'save_mepes') and self._args.save_mepes:
            output_dict = self.set_mepe_per_sample(loss_dict, example_dict, output_dict)

        # -------------------------------------------------------------
        # Return loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, output_dict, batch_size

    def set_mepe_per_sample(self, loss_dict, example_dict, output_dict):
        epe_dict = {}
        for six in range(len(example_dict['basedir'])):
            filename = os.path.join(example_dict['basedir'][six], example_dict['basename'][six])
            epe_dict[filename] = {'mepe': loss_dict['mepes'][six],
                                  'occ_f1': float(loss_dict['F1'][six].detach().cpu().numpy())}
        if not 'epe_dict' in output_dict:
            output_dict['epe_dict'] = epe_dict
        else:
            output_dict['epe_dict'].update(epe_dict)
        return output_dict

    def run(self, offset=0):

        with torch.no_grad():

            # ---------------------------------------
            # Tell model that we want to evaluate
            # ---------------------------------------
            self._model_and_loss.eval()

            # ---------------------------------------
            # Keep track of moving averages
            # ---------------------------------------
            moving_averages_dict = None

            # ---------------------------------------
            # Progress bar arguments
            # ---------------------------------------
            progressbar_args = {
                "iterable": self._loader,
                "desc": self._desc,
                "train": False,
                "offset": offset,
                "logging_on_update": False,
                "logging_on_close": True,
                "postfix": True
            }

            # ---------------------------------------
            # Perform evaluation steps
            # ---------------------------------------
            with create_progressbar(**progressbar_args) as progress:
                for example_dict in progress:

                    if self._args.white_list:
                        if len(example_dict['basedir']) > 1:
                            print("White list is supported with eval batch size 1 only")
                            exit(0)
                        filename = os.path.join(example_dict['basedir'][0], example_dict['basename'][0])
                        if filename not in self._args.white_list:
                            print("Skipping {}".format(filename))
                            continue
                        print("EVALUATING {}".format(filename))

                    # ---------------------------------------
                    # Perform forward evaluation step
                    # ---------------------------------------
                    loss_dict_per_step, output_dict, batch_size = self._step(example_dict)

                    # --------------------------------------------------------
                    # Save results
                    # --------------------------------------------------------
                    if self._save_output:
                        save_outputs(self._args, example_dict, output_dict)

                    if hasattr(self._args, 'save_mepes') and self._args.save_mepes:
                        self.samples_mepe.update(output_dict['epe_dict'])

                    # ---------------------------------------
                    # Convert loss dictionary to float
                    # ---------------------------------------
                    loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                    # --------------------------------------------------------
                    # Possibly initialize moving averages
                    # --------------------------------------------------------
                    if moving_averages_dict is None:
                        moving_averages_dict = {
                            key: MovingAverage() for key in loss_dict_per_step.keys()
                        }

                    # --------------------------------------------------------
                    # Add moving averages
                    # --------------------------------------------------------
                    for key, loss in loss_dict_per_step.items():
                        moving_averages_dict[key].add_average(loss, addcount=batch_size)

                    # view statistics in progress bar
                    progress_stats = format_moving_averages_as_progress_dict(
                        moving_averages_dict=moving_averages_dict,
                        moving_averages_postfix="_avg")

                    progress.set_postfix(progress_stats)

            # -------------------------------------------------------------
            # Record average losses
            # -------------------------------------------------------------
            avg_loss_dict = {key: ma.mean() for key, ma in moving_averages_dict.items()}

            # Save epe per sample
            if hasattr(self._args, 'save_mepes') and self._args.save_mepes:
                avg_loss_dict['epe_per_sample'] = self.samples_mepe
                model_name = os.path.basename(self._args.checkpoint)
                with open(os.path.join(self._args.save,
                                       "mepes_compare/{}.p".format(model_name)), 'wb') as f:
                    pickle.dump(self.samples_mepe, f)

            # -------------------------------------------------------------
            # Return average losses and output dictionary
            # -------------------------------------------------------------
            return avg_loss_dict


def exec_runtime(args,
                 checkpoint_saver,
                 model_and_loss,
                 optimizer,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 inference_loader,
                 training_augmentation,
                 validation_augmentation):
    # ----------------------------------------------------------------------------------------------
    # Validation schedulers are a bit special:
    # They want to be called with a validation loss..
    # ----------------------------------------------------------------------------------------------
    validation_scheduler = (lr_scheduler is not None and args.lr_scheduler == "ReduceLROnPlateau")

    # --------------------------------------------------------
    # Log some runtime info
    # --------------------------------------------------------
    with logger.LoggingBlock("Runtime", emph=True):
        logging.info("start_epoch: %i" % args.start_epoch)
        logging.info("total_epochs: %i" % args.total_epochs)

    # ---------------------------------------
    # Total progress bar arguments
    # ---------------------------------------
    progressbar_args = {
        "desc": "Progress",
        "initial": args.start_epoch - 1,
        "invert_iterations": True,
        "iterable": range(1, args.total_epochs + 1),
        "logging_on_close": True,
        "logging_on_update": True,
        "postfix": False,
        "unit": "ep"
    }

    # --------------------------------------------------------
    # Total progress bar
    # --------------------------------------------------------
    print(''), logging.logbook('')
    total_progress = create_progressbar(**progressbar_args)
    print("\n")

    # --------------------------------------------------------
    # Remember validation loss
    # --------------------------------------------------------
    best_validation_loss = float("inf") if args.validation_key_minimize else -float("inf")
    store_as_best = False

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        with logger.LoggingBlock("Epoch %i/%i" % (epoch, args.total_epochs), emph=True):

            # --------------------------------------------------------
            # Update standard learning scheduler
            # --------------------------------------------------------
            if lr_scheduler is not None and not validation_scheduler:
                lr_scheduler.step(epoch)

            # --------------------------------------------------------
            # Always report learning rate
            # --------------------------------------------------------
            if lr_scheduler is None:
                logging.info("lr: %s" % format_learning_rate(args.optimizer_lr))
            else:
                logging.info("lr: %s" % format_learning_rate(lr_scheduler.get_lr()))

            # -------------------------------------------
            # Create and run a training epoch
            # -------------------------------------------
            if train_loader is not None:
                avg_loss_dict = TrainingEpoch(
                    args,
                    desc="   Train",
                    model_and_loss=model_and_loss,
                    optimizer=optimizer,
                    loader=train_loader,
                    augmentation=training_augmentation).run()

            # -------------------------------------------
            # Create and run a validation epoch
            # -------------------------------------------
            if validation_loader is not None:

                # ---------------------------------------------------
                # Construct holistic recorder for epoch
                # ---------------------------------------------------
                avg_loss_dict = EvaluationEpoch(
                    args,
                    desc="Validate",
                    model_and_loss=model_and_loss,
                    loader=validation_loader,
                    augmentation=validation_augmentation).run()

                # ----------------------------------------------------------------
                # Evaluate whether this is the best validation_loss
                # ----------------------------------------------------------------
                validation_loss = avg_loss_dict[args.validation_key]
                if args.validation_key_minimize:
                    store_as_best = validation_loss < best_validation_loss
                else:
                    store_as_best = validation_loss > best_validation_loss
                if store_as_best:
                    best_validation_loss = validation_loss

                # ----------------------------------------------------------------
                # Update validation scheduler, if one is in place
                # ----------------------------------------------------------------
                if lr_scheduler is not None and validation_scheduler:
                    lr_scheduler.step(validation_loss, epoch=epoch)

            # ----------------------------------------------------------------
            # Also show best loss on total_progress
            # ----------------------------------------------------------------
            total_progress_stats = {
                "best_" + args.validation_key + "_avg": "%1.4f" % best_validation_loss
            }
            total_progress.set_postfix(total_progress_stats)

            # ----------------------------------------------------------------
            # Bump total progress
            # ----------------------------------------------------------------
            total_progress.update()
            print('')

            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None:
                if args.max_save and (args.min_save <= epoch <= args.max_save):
                    checkpoint_saver.save_latest(directory=args.save,
                                                 model_and_loss=model_and_loss,
                                                 stats_dict=dict(avg_loss_dict,
                                                                 epoch=epoch),
                                                 suffix="_epoch_{}".format(epoch))

                checkpoint_saver.save_latest(
                    directory=args.save,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch),
                    store_as_best=store_as_best)

            # ----------------------------------------------------------------
            # Vertical space between epochs
            # ----------------------------------------------------------------
            print(''), logging.logbook('')

    # ----------------------------------------------------------------
    # Finish
    # ----------------------------------------------------------------
    total_progress.close()
    logging.info("Finished.")

    return avg_loss_dict
