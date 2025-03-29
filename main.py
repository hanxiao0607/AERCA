import sys
import os
import logging
import argparse

from datasets import linear, lotka_volterra, lorenz96, swat, nonlinear, msds
from args import linear_args, lotka_volterra_args, lorenz96_args, swat_args, msds_args, nonlinear_args
from models import aerca
from utils import utils
import warnings
warnings.filterwarnings("ignore")

def main(argv):
    """
    Main function to run the AERCA model on a specified dataset.

    The script supports multiple datasets. It selects the appropriate dataset class,
    argument parser, and log file based on the provided --dataset_name argument.
    If preprocessing_data is set to 1, the dataset is generated and saved; otherwise,
    the existing data is loaded.

    After setting the random seed and loading the data, the AERCA model is instantiated
    with common parameters and then trained and tested accordingly.

    Args:
        argv (list): List of command-line arguments.
    """
    # Preliminary parsing: retrieve the dataset name.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--dataset_name',
        type=str,
        default='linear',
        help='Name of the dataset to run. Options: linear, lotka_volterra, lorenz96, msds, swat, nonlinear'
    )
    pre_args, remaining_args = pre_parser.parse_known_args(argv[1:])
    dataset_name = pre_args.dataset_name.lower()

    # Map dataset names to their configuration: argument parser, dataset class, log file, and slicing flag.
    dataset_mapping = {
        "linear": {
            "args": linear_args.create_arg_parser,
            "dataset_class": linear.Linear,
            "log_file": "linear.log",
            "use_slice": True
        },
        "lotka_volterra": {
            "args": lotka_volterra_args.create_arg_parser,
            "dataset_class": lotka_volterra.LotkaVolterra,
            "log_file": "lotka_volterra.log",
            "use_slice": True
        },
        "lorenz96": {
            "args": lorenz96_args.create_arg_parser,
            "dataset_class": lorenz96.Lorenz96,
            "log_file": "lorenz96.log",
            "use_slice": True
        },
        "msds": {
            "args": msds_args.create_arg_parser,
            "dataset_class": msds.MSDS,
            "log_file": "msds.log",
            "use_slice": False
        },
        "swat": {
            "args": swat_args.create_arg_parser,
            "dataset_class": swat.SWaT,
            "log_file": "swat.log",
            "use_slice": False
        },
        "nonlinear": {
            "args": nonlinear_args.create_arg_parser,
            "dataset_class": nonlinear.Nonlinear,
            "log_file": "nonlinear.log",
            "use_slice": True
        }
    }

    # Ensure the specified dataset is recognized.
    if dataset_name not in dataset_mapping:
        print("Dataset '{}' not recognized. Available options are: {}"
              .format(dataset_name, list(dataset_mapping.keys())))
        sys.exit(1)

    mapping = dataset_mapping[dataset_name]

    # Set up logging: create a logs directory relative to the current working directory if it doesn't exist.
    logging_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_file_path = os.path.join(logging_dir, mapping["log_file"])
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse the remaining command-line arguments using the dataset-specific argument parser.
    parser = mapping["args"]()
    args, unknown = parser.parse_known_args(remaining_args)
    options = vars(args)

    # Set the random seed for reproducibility.
    utils.set_seed(options['seed'])
    print('Set seed: {}'.format(options['seed']))

    # Instantiate the dataset class and generate or load data based on the preprocessing flag.
    data_class = mapping["dataset_class"](options)
    if options['preprocessing_data'] == 1:
        print('Preprocessing data: generating and saving new data...')
        data_class.generate_example()
        data_class.save_data()
    else:
        print('Loading existing data...')
        data_class.load_data()

    # Instantiate the AERCA model using the common set of parameters.
    aerca_model = aerca.AERCA(
        num_vars=options['num_vars'],
        hidden_layer_size=options['hidden_layer_size'],
        num_hidden_layers=options['num_hidden_layers'],
        device=options['device'],
        window_size=options['window_size'],
        stride=options['stride'],
        encoder_gamma=options['encoder_gamma'],
        decoder_gamma=options['decoder_gamma'],
        encoder_lambda=options['encoder_lambda'],
        decoder_lambda=options['decoder_lambda'],
        beta=options['beta'],
        lr=options['lr'],
        epochs=options['epochs'],
        recon_threshold=options['recon_threshold'],
        data_name=options['dataset_name'],
        causal_quantile=options['causal_quantile'],
        root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
        root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
        risk=options['risk'],
        initial_level=options['initial_level'],
        num_candidates=options['num_candidates']
    )

    # Training phase: train the model if enabled.
    if options['training_aerca']:
        # Use slicing for training data if required by the dataset configuration.
        if mapping["use_slice"]:
            training_data = data_class.data_dict['x_n_list'][:options['training_size']]
        else:
            training_data = data_class.data_dict['x_n_list']
        print('Start training AERCA model...')
        aerca_model._training(training_data)
        print('Done training')

    # Testing phase for causal discovery (applies only if slicing is used).
    if mapping["use_slice"]:
        test_causal = data_class.data_dict['x_n_list'][options['training_size']:]
        print('Start testing AERCA model for causal discovery...')
        aerca_model._testing_causal_discover(test_causal, data_class.data_dict['causal_struct'])
        print('Done testing for causal discovery')

    # Testing phase for root cause analysis.
    if mapping["use_slice"]:
        test_x_ab = data_class.data_dict['x_ab_list'][options['training_size']:]
        test_label = data_class.data_dict['label_list'][options['training_size']:]
    else:
        test_x_ab = data_class.data_dict['x_ab_list']
        test_label = data_class.data_dict['label_list']
    print('Start testing AERCA model for root cause analysis...')
    aerca_model._testing_root_cause(test_x_ab, test_label)
    print('Done testing for root cause analysis')


    print('done')

if __name__ == '__main__':
    main(sys.argv)
