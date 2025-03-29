import argparse
import os

def create_arg_parser():
    """
    Creates and returns the argument parser for the Nonlinear dataset.

    Returns:
        argparse.ArgumentParser: The argument parser for the Nonlinear dataset.
    """
    parser = argparse.ArgumentParser(description='Nonlinear')

    # Dataset arguments
    parser.add_argument('--T', type=int, default=500, help='Length of the time series (default: 500)')
    parser.add_argument('--training_size', type=int, default=10, help='Size of the training set (default: 10)')
    parser.add_argument('--testing_size', type=int, default=100, help='Size of the testing set (default: 100)')
    parser.add_argument('--num_vars', type=int, default=6, help='Number of variables (default: 6)')
    parser.add_argument('--preprocessing_data', type=int, default=1, help='Flag for preprocessing data (default: 1)')
    parser.add_argument('--adlength', type=int, default=1, help='Ad length (default: 1)')
    parser.add_argument('--adtype', type=str, default='non_causal', help='Ad type (default: non_causal)')
    parser.add_argument('--mul', type=int, default=10, help='Multiplier (default: 10)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets', 'nonlinear'), help='Data directory (default: ./datasets/nonlinear)')
    parser.add_argument('--causal_quantile', type=float, default=0.80, help='Causal quantile (default: 0.80)')
    parser.add_argument('--m', type=int, default=7, help='Parameter m (default: 7)')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Noise scale (default: 1.0)')
    parser.add_argument('--dependent_features', type=int, default=0, help='Flag for dependent features (default: 0)')

    # Meta arguments
    parser.add_argument('--seed', type=int, default=36, help='Random seed (default: 36)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--dataset_name', type=str, default='nonlinear', help='Dataset name (default: nonlinear)')

    # AERCA arguments
    parser.add_argument('--window_size', type=int, default=5, help='Window size (default: 5)')
    parser.add_argument('--stride', type=int, default=1, help='Stride (default: 1)')
    parser.add_argument('--encoder_alpha', type=float, default=0.5, help='Encoder alpha (default: 0.5)')
    parser.add_argument('--decoder_alpha', type=float, default=0.5, help='Decoder alpha (default: 0.5)')
    parser.add_argument('--encoder_gamma', type=float, default=0.5, help='Encoder gamma (default: 0.5)')
    parser.add_argument('--decoder_gamma', type=float, default=0.5, help='Decoder gamma (default: 0.5)')
    parser.add_argument('--encoder_lambda', type=float, default=0.5, help='Encoder lambda (default: 0.5)')
    parser.add_argument('--decoder_lambda', type=float, default=0.5, help='Decoder lambda (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs (default: 5000)')
    parser.add_argument('--hidden_layer_size', type=int, default=50, help='Hidden layer size (default: 50)')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='Number of hidden layers (default: 8)')
    parser.add_argument('--recon_threshold', type=float, default=0.95, help='Reconstruction threshold (default: 0.95)')
    parser.add_argument('--root_cause_threshold_encoder', type=float, default=0.99, help='Root cause threshold for encoder (default: 0.99)')
    parser.add_argument('--root_cause_threshold_decoder', type=float, default=0.99, help='Root cause threshold for decoder (default: 0.99)')
    parser.add_argument('--training_aerca', type=int, default=1, help='Flag for training AERCA (default: 1)')
    parser.add_argument('--initial_z_score', type=float, default=3.0, help='Initial Z-score (default: 3.0)')
    parser.add_argument('--risk', type=float, default=1e-2, help='Risk (default: 1e-2)')
    parser.add_argument('--initial_level', type=float, default=0.98, help='Initial level (default: 0.98)')
    parser.add_argument('--num_candidates', type=int, default=100, help='Number of candidates (default: 100)')

    return parser

if __name__ == "__main__":
    try:
        arg_parser = create_arg_parser()
        args = arg_parser.parse_args()
        print(args)
    except Exception as e:
        print(f"Error parsing arguments: {e}")