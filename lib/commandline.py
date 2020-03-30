## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import os
import argparse
import colorama
import inspect
import sys
import yaml
import torch
import datasets
import models
import logging

from lib import losses, augmentations, optimizers as optim
from utils import tools, logger


def _get_type_from_arg(arg):
    if isinstance(arg, bool):
        return tools.str2bool
    else:
        return type(arg)


def _add_arguments_for_module(parser,
                              module,
                              name,
                              default_class,
                              add_class_argument=True,  # whether to add class choice as argument
                              include_classes="*",
                              exclude_classes=[],
                              exclude_params=["self", "args"],
                              param_defaults={},  # allows to overwrite any default param
                              forced_default_types={},  # allows to set types for known arguments
                              unknown_default_types={}):  # allows to set types for unknown arguments

    # -------------------------------------------------------------------------
    # Determine possible choices from class names in module, possibly apply include/exclude filters
    # -------------------------------------------------------------------------
    module_dict = tools.module_classes_to_dict(
        module, include_classes=include_classes, exclude_classes=exclude_classes)

    # -------------------------------------------------------------------------
    # Parse known arguments to determine choice for argument name
    # -------------------------------------------------------------------------
    if add_class_argument:
        parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = parser.parse_known_args(sys.argv[1:])[0]
    else:
        # build a temporary parser, and do not add the class as argument
        tmp_parser = argparse.ArgumentParser()
        tmp_parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = tmp_parser.parse_known_args(sys.argv[1:])[0]

    class_name = vars(known_args)[name]

    # -------------------------------------------------------------------------
    # If class is None, there is no point in trying to parse further arguments
    # -------------------------------------------------------------------------
    if class_name is None:
        return

    # -------------------------------------------------------------------------
    # Get constructor of that argument choice
    # -------------------------------------------------------------------------
    class_constructor = module_dict[class_name]

    # -------------------------------------------------------------------------
    # Determine constructor argument names and defaults
    # -------------------------------------------------------------------------
    try:
        argspec = inspect.getargspec(class_constructor.__init__)
        argspec_defaults = argspec.defaults if argspec.defaults is not None else []
        full_args = argspec.args
        default_args_dict = dict(zip(argspec.args[-len(argspec_defaults):], argspec_defaults))
    except TypeError:
        print(argspec)
        print(argspec.defaults)
        raise ValueError("unknown_default_types should be adjusted for module: '%s.py'" % name)

    # -------------------------------------------------------------------------
    # Add sub_arguments
    # -------------------------------------------------------------------------
    for argname in full_args:

        # ---------------------------------------------------------------------
        # Skip
        # ---------------------------------------------------------------------
        if argname in exclude_params:
            continue

        # ---------------------------------------------------------------------
        # Sub argument name
        # ---------------------------------------------------------------------
        sub_arg_name = "%s_%s" % (name, argname)

        # ---------------------------------------------------------------------
        # If a default argument is given, take that one
        # ---------------------------------------------------------------------
        if argname in param_defaults.keys():
            parser.add_argument(
                "--%s" % sub_arg_name,
                type=_get_type_from_arg(param_defaults[argname]),
                default=param_defaults[argname])

        # ---------------------------------------------------------------------
        # If a default parameter can be inferred from the module, pick that one
        # ---------------------------------------------------------------------
        elif argname in default_args_dict.keys():

            # -----------------------------------------------------------------
            # Check for forced default types
            # -----------------------------------------------------------------
            if argname in forced_default_types.keys():
                argtype = forced_default_types[argname]
            else:
                argtype = _get_type_from_arg(default_args_dict[argname])
            parser.add_argument(
                "--%s" % sub_arg_name, type=argtype, default=default_args_dict[argname])

        # ---------------------------------------------------------------------
        # Take from the unkowns list
        # ---------------------------------------------------------------------
        elif argname in unknown_default_types.keys():
            parser.add_argument("--%s" % sub_arg_name, type=unknown_default_types[argname])

        else:
            raise ValueError(
                "Do not know how to handle argument '%s' for class '%s'" % (argname, name))


def _add_special_arguments(parser):
    # -------------------------------------------------------------------------
    # Known arguments so far
    # -------------------------------------------------------------------------
    known_args = vars(parser.parse_known_args(sys.argv[1:])[0])

    # -------------------------------------------------------------------------
    # Add special arguments for training
    # -------------------------------------------------------------------------
    training_loss = known_args["training_loss"]
    if training_loss is not None:
        parser.add_argument("--training_key", type=str, default="total_loss")

    # -------------------------------------------------------------------------
    # Add special arguments for validation
    # -------------------------------------------------------------------------
    validation_loss = known_args["validation_loss"]
    if validation_loss is not None:
        parser.add_argument("--validation_key", type=str, default="total_loss")
        parser.add_argument("--validation_key_minimize", type=tools.str2bool, default=True)

    # -------------------------------------------------------------------------
    # Add special arguments for checkpoints
    # -------------------------------------------------------------------------
    checkpoint = known_args["checkpoint"]
    if checkpoint is not None:
        parser.add_argument(
            "--checkpoint_mode", type=str, default="resume_from_latest",
            choices=["resume_from_latest", "resume_from_best"])

        parser.add_argument(
            "--checkpoint_include_params", type=tools.str2list, default="[*]")
        parser.add_argument(
            "--checkpoint_exclude_params", type=tools.str2list, default="[]")

    # -------------------------------------------------------------------------
    # Add special arguments for optimizer groups
    # -------------------------------------------------------------------------
    parser.add_argument("--optimizer_group", action="append", type=tools.str2dict, default=None)


def override_args_from_yaml(args, yaml_conf=None,
                            override_default_conf=False):
    """

    :param args:
    :param yaml_conf:
    :param override_default_conf:
    :return:
    """
    if override_default_conf:
        with open('config/latest.yaml', 'w') as ymlfile:
            yaml.dump(args.__dict__, ymlfile)

    config_file = yaml_conf if yaml_conf else args.config_file

    if config_file:

        if not os.path.exists(config_file):
            config_file = os.path.join(args.save, config_file)

        with open(config_file, 'r') as ymlfile:
            yaml_settings = yaml.load(ymlfile)

            # Go over yaml settings and add to args
            for k, v in yaml_settings.items():
                if v is not None:
                    if type(v) is str:
                        exec('args.{}="{}"'.format(k, v))
                    else:
                        exec('args.{}={}'.format(k, v))

    if args.save_conf:
        yaml_path = os.path.join(args.save, 'config.yaml')
        if not os.path.isdir(args.save):
            os.mkdir(args.save)
        with open(yaml_path, 'w') as ymlfile:
            yaml.dump(args.__dict__, ymlfile)

    return args


def _parse_arguments(yaml_conf=None):
    # -------------------------------------------------------------------------
    # Argument parser and shortcut function to add arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    # -------------------------------------------------------------------------
    # Standard arguments
    # -------------------------------------------------------------------------
    add("--batch_size", type=int, default=4)
    add("--batch_size_val", type=int, default=4)
    add("--checkpoint", type=tools.str2str_or_none, default=None)
    add("--config_file", '-f', type=str, default='PLEASE_PICK_A_CONFIG_FILE')
    add("--cuda", type=tools.str2bool, default=True)
    add("--devices", '-d', type=str, default='0')
    add("--evaluation", type=tools.str2bool, default=False)
    add("--name", default="run", type=str)
    add("--num_workers", type=int, default=4)
    add("--save", "-s", default="config/", type=str)
    add("--seed", type=int, default=1)
    add("--start_epoch", type=int, default=1)
    add("--total_epochs", type=int, default=10)
    add("--min_save", type=int, default=-1)
    add("--max_save", type=int, default=-1)
    add("--save_conf", type=tools.str2bool, default=True)
    add("--save_result_path_name", default="", type=str)
    add("--save_result_img", type=tools.str2bool, default=False)
    add("--save_result_occ", type=tools.str2bool, default=False)
    add("--save_result_flo", type=tools.str2bool, default=False)
    add("--save_result_png", type=tools.str2bool, default=False)
    add("--save_result_bidirection", type=tools.str2bool, default=False)
    add("--num_iters", type=int, default=1)
    add("--get_tmp_results", default="", type=str)
    add("--show_aug", type=tools.str2bool, default=False)
    add("--crop_min_ratio", default=0.95, type=float)
    add("--crop_max_ratio", default=1, type=float)


    # -------------------------------------------------------------------------
    # Arguments inferred from losses
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        losses,
        name="training_loss",
        default_class=None,
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    _add_arguments_for_module(
        parser,
        losses,
        name="validation_loss",
        default_class=None,
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from models
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        models,
        name="model",
        default_class="IRR_PWC_V2",
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from augmentations for training
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        augmentations,
        name="training_augmentation",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"],
        forced_default_types={"crop": tools.str2intlist})

    # -------------------------------------------------------------------------
    # Arguments inferred from augmentations for validation
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        augmentations,
        name="validation_augmentation",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from datasets for training
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        datasets,
        name="training_dataset",
        default_class=None,
        exclude_params=["self", "args", "is_cropped"],
        exclude_classes=["_*"],
        unknown_default_types={"root": str})

    # -------------------------------------------------------------------------
    # Arguments inferred from datasets for validation
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        datasets,
        name="validation_dataset",
        default_class=None,
        exclude_params=["self", "args", "is_cropped"],
        exclude_classes=["_*"],
        unknown_default_types={"root": str})

    # -------------------------------------------------------------------------
    # Arguments inferred from PyTorch optimizers
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        optim,
        name="optimizer",
        default_class="Adam",
        exclude_classes=["_*", "Optimizer", "constructor"],
        exclude_params=["self", "args", "params"],
        forced_default_types={"lr": float,
                              "momentum": float,
                              "dampening": float,
                              "weight_decay": float,
                              "nesterov": tools.str2bool})

    # -------------------------------------------------------------------------
    # Arguments inferred from PyTorch lr schedulers
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        torch.optim.lr_scheduler,
        name="lr_scheduler",
        default_class=None,
        exclude_classes=["_*", "Optimizer"],
        exclude_params=["self", "args", "optimizer"],
        unknown_default_types={"T_max": int,
                               "lr_lambda": str,
                               "step_size": int,
                               "milestones": tools.str2intlist,
                               "gamma": float})

    # -------------------------------------------------------------------------
    # Special arguments
    # -------------------------------------------------------------------------
    _add_special_arguments(parser)

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Override from yaml if provided
    # -------------------------------------------------------------------------
    args = override_args_from_yaml(args, yaml_conf)

    # -------------------------------------------------------------------------
    # Parse default arguments from a dummy commandline not specifying any args
    # -------------------------------------------------------------------------
    defaults = vars(parser.parse_known_args(['--dummy'])[0])

    # -------------------------------------------------------------------------
    # Consistency checks
    # -------------------------------------------------------------------------
    args.cuda = args.cuda and torch.cuda.is_available()

    return args, defaults


def postprocess_args(args):
    # ----------------------------------------------------------------------------
    # Get appropriate class constructors from modules
    # ----------------------------------------------------------------------------
    args.model_class = tools.module_classes_to_dict(models)[args.model]

    if args.optimizer is not None:
        optimizer_classes = tools.module_classes_to_dict(optim)
        args.optimizer_class = optimizer_classes[args.optimizer]

    if args.training_loss is not None:
        loss_classes = tools.module_classes_to_dict(losses)
        args.training_loss_class = loss_classes[args.training_loss]

    if args.validation_loss is not None:
        loss_classes = tools.module_classes_to_dict(losses)
        args.validation_loss_class = loss_classes[args.validation_loss]

    if args.lr_scheduler is not None:
        scheduler_classes = tools.module_classes_to_dict(torch.optim.lr_scheduler)
        args.lr_scheduler_class = scheduler_classes[args.lr_scheduler]

    if args.training_dataset is not None:
        dataset_classes = tools.module_classes_to_dict(datasets)
        args.training_dataset_class = dataset_classes[args.training_dataset]

    if args.validation_dataset is not None:
        dataset_classes = tools.module_classes_to_dict(datasets)
        args.validation_dataset_class = dataset_classes[args.validation_dataset]

    if args.training_augmentation is not None:
        augmentation_classes = tools.module_classes_to_dict(augmentations)
        args.training_augmentation_class = augmentation_classes[args.training_augmentation]

    if args.validation_augmentation is not None:
        augmentation_classes = tools.module_classes_to_dict(augmentations)
        args.validation_augmentation_class = augmentation_classes[args.validation_augmentation]

    return args


def setup_logging_and_parse_arguments(blocktitle, yaml_conf=None):
    # ----------------------------------------------------------------------------
    # Get parse commandline and default arguments
    # ----------------------------------------------------------------------------
    args, defaults = _parse_arguments(yaml_conf=yaml_conf)

    # ----------------------------------------------------------------------------
    # Setup logbook before everything else
    # ----------------------------------------------------------------------------
    logger.configure_logging(os.path.join(args.save, 'logbook.txt'))

    # ----------------------------------------------------------------------------
    # Write arguments to file, as txt
    # ----------------------------------------------------------------------------
    tools.write_dictionary_to_file(
        sorted(vars(args).items()),
        filename=os.path.join(args.save, 'args.txt'))

    # ----------------------------------------------------------------------------
    # Log arguments
    # ----------------------------------------------------------------------------
    with logger.LoggingBlock(blocktitle, emph=True):
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if (argument in defaults and value == defaults[argument]) else colorama.Fore.CYAN
            logging.info('{}{}: {}{}'.format(color, argument, value, reset))

    # ----------------------------------------------------------------------------
    # Postprocess
    # ----------------------------------------------------------------------------
    args = postprocess_args(args)

    return args
