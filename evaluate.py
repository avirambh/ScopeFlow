import os
from lib import commandline, validation
from utils import constants

# Global override
YAML = None  #'config/evaluation/eval_template_sintel.yaml'
SAVE_FLOW = False
SAVE_PNG = False
CREATE_BUNDLE = False
EVAL_DATASETS = constants.SINTEL_VAL_DATASETS  # Defaults to Sintel
FALLBACK_DEVICES_TO_USE = '0,1'

if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = FALLBACK_DEVICES_TO_USE


def set_eval_overrides(args):
    args.SAVE_FLOW = SAVE_FLOW or args.save_result_flo or args.save_result_occ
    args.SAVE_PNG = SAVE_PNG or args.save_result_png
    args.CREATE_BUNDLE = CREATE_BUNDLE
    if hasattr(args, 'eval_train_val') and args.eval_train_val:
        args.EVAL_DATASETS = EVAL_DATASETS
    else:
        args.EVAL_DATASETS = [args.validation_dataset]
    return args


def evaluate():
    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Parse commandline arguments
    args = commandline.setup_logging_and_parse_arguments(blocktitle="Commandline Arguments",
                                                         yaml_conf=YAML)

    args = set_eval_overrides(args)
    validation.validate(args)


if __name__ == '__main__':
    evaluate()
