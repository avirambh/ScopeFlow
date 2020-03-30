from __future__ import absolute_import, division, print_function

import os
import torch
import logging
from lib import runtime, configuration as config
from utils import logger


def train_and_val(args):

    # Set forced devices
    if os.environ['CUDA_VISIBLE_DEVICES']:
        args.devices = os.environ['CUDA_VISIBLE_DEVICES']

    # Set random seed, possibly on Cuda
    config.configure_random_seed(args)

    # DataLoader
    train_loader, validation_loader, inference_loader = config.configure_data_loaders(args)
    success = any(loader is not None for loader in [train_loader, validation_loader, inference_loader])
    if not success:
        logging.info("No dataset could be loaded successfully. Please check dataset paths!")
        quit()

    # Configure data augmentation
    training_augmentation, validation_augmentation = config.configure_runtime_augmentations(args)

    # Configure model and loss
    model_and_loss = config.configure_model_and_loss(args)

    # Resume from checkpoint if available
    checkpoint_saver, checkpoint_stats = config.configure_checkpoint_saver(args, model_and_loss)

    # Checkpoint and save directory
    with logger.LoggingBlock("Save Directory", emph=True):
        logging.info("Save directory: %s" % args.save)
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # Multi-GPU automation
    with logger.LoggingBlock("GPU allocation", emph=True):
        if 'devices' in args.__dict__:
            gpus = str(args.devices)
            num_gpus = len(gpus.split(','))
            if num_gpus > 1:
                if torch.cuda.device_count() > 1:
                    logging.info("Let's use {} GPUs! GPUS: {}".format(num_gpus, gpus))
                    model_and_loss._model = torch.nn.DataParallel(model_and_loss._model)
                else:
                    logging.info("Let's use {} GPU! GPU: {}".format(num_gpus, gpus))
            else:
                logging.info("Using device number: {}".format(args.devices))

    # Configure optimizer
    optimizer = config.configure_optimizer(args, model_and_loss)

    # Configure learning rate
    lr_scheduler = config.configure_lr_scheduler(args, optimizer)

    # If this is just an evaluation: overwrite savers and epochs
    if args.evaluation:
        args.start_epoch = 1
        args.total_epochs = 1
        train_loader = None
        checkpoint_saver = None
        optimizer = None
        lr_scheduler = None

    # Cuda optimization
    if args.cuda:
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True  # A larger CUDA footprint
        pass

    # Kickoff training, validation and/or testing
    results = runtime.exec_runtime(
        args,
        checkpoint_saver=checkpoint_saver,
        model_and_loss=model_and_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        inference_loader=inference_loader,
        training_augmentation=training_augmentation,
        validation_augmentation=validation_augmentation)

    # Shut down logging to allow consecutive trainings
    logging.shutdown(logging._handlerList)

    return results


def main():
    train_and_val()


if __name__ == "__main__":
    main()

