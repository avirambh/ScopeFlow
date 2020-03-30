import os
from lib.pipeline_wrapper import train_and_val
from lib.commandline import postprocess_args
from utils.constants import results_keys, results_mapping, \
    OUTPUT_PATH, BUNDLES_DIR, DEFAULT_SINTEL_DIR


def print_formatted_results(results):
    print('Method, ' + ', '.join(results_keys))

    for m, res in results.items():

        # Fill missing keys
        for k in results_keys:
            dataset = results_mapping[k][0]
            metric = results_mapping[k][1]
            if dataset not in res:
                res[dataset] = {}
            if metric not in res[dataset]:
                res[dataset][metric] = -1

        # Print results for model
        format_output = "{}" + ", {:.5f}" * len(results_keys)
        print(format_output.format(m,
                                   *(res[results_mapping[k][0]][results_mapping[k][1]]
                                     for k in results_keys)))


def create_sintel_bundle(model_out_folder, model_bname='output',
                         sintel_path=DEFAULT_SINTEL_DIR,
                         bundles_dir=BUNDLES_DIR):
    print("Creating sintel bundle for model {}".format(model_bname))
    model_out_folder = '{}/flo/data/sintel/test/'.format(model_out_folder)
    clean_out_path = '{}/clean'.format(model_out_folder)
    final_out_path = '{}/final'.format(model_out_folder)
    lzma_out_path = os.path.join(bundles_dir, '{}_bundle.lzma'.format(model_bname))
    if not os.path.isdir(bundles_dir):
        os.mkdir(bundles_dir)

    print("Creating bundle!\nClean path: {}, Final path: {}".format(clean_out_path,
                                                                    final_out_path))

    bundler_path = os.path.join(sintel_path, 'bundler/linux-x64/bundler')
    if not os.path.isdir(bundles_dir):
        print("Creating bundles directory")
        os.makedirs(bundles_dir)

    os.system(' '.join([bundler_path, clean_out_path, final_out_path, lzma_out_path]))
    print("Bundle is saved at: {}.\nGood luck!".format(lzma_out_path))


def set_val_args(tmp_args, args, d, model_out_folder):
    tmp_args['validation_dataset'] = d

    if args.SAVE_FLOW or args.SAVE_PNG:
        tmp_args['save_result_path_name'] = model_out_folder
        tmp_args['save'] = model_out_folder
    else:
        tmp_args['save_result_path_name'] = ''
        tmp_args['save'] = OUTPUT_PATH

    tmp_args['save_result_flo'] = args.SAVE_FLOW
    tmp_args['save_result_occ'] = args.SAVE_FLOW
    tmp_args['save_result_img'] = args.SAVE_FLOW or args.SAVE_PNG
    tmp_args['save_result_png'] = args.SAVE_PNG
    return tmp_args


def validate(args):
    results = {}
    skipped_models = []

    #print("Testing models: {}".format(args.checkpoint))
    #print("Testing datasets: {}".format(args.validation_dataset))

    m = args.checkpoint
    best_model = os.path.join(m, 'checkpoint_best.ckpt')
    if not os.path.exists(best_model):
        print("Skipping {} - no best module".format(m))
        skipped_models.append(m)
        exit(1)

    # Get model specific args
    model_bname = os.path.basename(m)  # Assume directory
    model_out_folder = os.path.join(OUTPUT_PATH, model_bname)

    results[model_bname] = {}
    for d in args.EVAL_DATASETS:
        tmp_args = {}
        tmp_args = set_val_args(tmp_args, args, d, model_out_folder)
        args.__dict__.update(tmp_args)
        args = postprocess_args(args)
        res = train_and_val(args)
        results[model_bname][d] = res
        print("Result: {}".format(res))

    print("\nResults so far:")
    print_formatted_results(results)

    if args.CREATE_BUNDLE:
        create_sintel_bundle(model_out_folder, model_bname)


if __name__ == '__main__':
    pass
