import argparse

parser = argparse.ArgumentParser(description='Description of your program')

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Path to input file', required=True)
parser.add_argument('-o', '--output', type=str, help='Path to output file', required=True)
parser.add_argument('-e', '--epoch', type=int, help='Number of epoch in federated learning', required=True)
parser.add_argument('-c', '--client_epoch', type=int, help='Number of epoch in the local training process of client', required=True)
parser.add_argument('-s', '--split', type=str, choices=['even', 'bias'], help='Strategy to split input to clients', required=True)

# Parse the arguments
args = parser.parse_args()