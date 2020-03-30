import os
from lib import commandline, pipeline_wrapper

# Configure devices
YAML = None  #'config/training/sintel_ft.yaml'
FALLBACK_DEVICES_TO_USE = '0,1'

if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = FALLBACK_DEVICES_TO_USE


def train():
    # Change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Parse commandline arguments
    args = commandline.setup_logging_and_parse_arguments(blocktitle="Commandline Arguments",
                                                         yaml_conf=YAML)

    # Run training
    pipeline_wrapper.train_and_val(args)


if __name__ == '__main__':
    train()
