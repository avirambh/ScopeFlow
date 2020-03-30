import os


# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
YAMLS_DIR = os.path.join(ROOT_DIR, 'config/evaluation/')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments/')

DEFAULT_SINTEL_DIR = '../data/sintel'
DEFAULT_KITTI_DIR = '../data/kitticomb'

SINTEL_VAL_DATASETS = ['SintelTrainingFinalValid', 'SintelTrainingCleanValid']
SINTEL_TEST_DATASETS = ['SintelTestClean', 'SintelTestFinal']
DEFAULT_YAMLS_PATH = os.path.join(YAMLS_DIR, 'eval_template_sintel.yaml')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output/')
BUNDLES_DIR = os.path.join(OUTPUT_PATH, 'bundles/')

# DICTS
results_keys = ['clean_val', 'clean_val_occ', 'clean_val_no_occ',
                'final_val', 'final_val_occ', 'final_val_no_occ',
                'clean_F1', 'final_F1']

results_mapping = {'clean_val': ['SintelTrainingCleanValid', 'epe'],
                   'clean_val_occ': ['SintelTrainingCleanValid', 'mepes_occ'],
                   'clean_val_no_occ': ['SintelTrainingCleanValid', 'mepes_no_occ'],
                   'final_val': ['SintelTrainingFinalValid', 'epe'],
                   'final_val_occ': ['SintelTrainingFinalValid', 'mepes_occ'],
                   'final_val_no_occ': ['SintelTrainingFinalValid', 'mepes_no_occ'],
                   'clean_F1': ['SintelTrainingCleanValid', 'F1'],
                   'final_F1': ['SintelTrainingFinalValid', 'F1']}

