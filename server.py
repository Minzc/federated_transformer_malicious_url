from collections import OrderedDict
from datasets import Features, Value, ClassLabel, load_dataset, Dataset, concatenate_datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import DatasetDict
from evaluate import load as load_metric
from transformers import AdamW
from transformers import AutoModelForSequenceClassification 
import flwr as fl
from typing import List, Tuple, Dict
from flwr.common import Metrics
import gc
from wordcloud import WordCloud
import tqdm
from flwr.common.parameter import parameters_to_ndarrays
import matplotlib.pyplot as plt
import json
from torch import nn
from util import weighted_average, get_evaluate_fn, init_model, NUM_LABELS, load_raw_data, prepare_train_test_iid, prepare_train_test_noniid, process_data
from client import MalURLClient
import torch
from typing import List, Dict, Any, Callable

def get_client_fn(client_dataloaders: List[Dict[str, Any]], net: torch.nn.Module) -> Callable[[str], Any]:
    """
    Return the function to create a client.

    Parameters
    ----------
    client_dataloaders : List[Dict[str, Any]]
        A list of dictionaries containing the data loaders for the training and validation data of all clients.
    net : torch.nn.Module
        The neural network to use for the clients.

    Returns
    -------
    Callable[[str], Any]
        A function that creates a client instance for the given client ID.
    """
    def client_fn(cid: str) -> Any:
        """
        Create a client instance for the given client ID.

        Parameters
        ----------
        cid: str
            The client ID.

        Returns
        -------
        Any: MalURLClient
            A client instance for the given client ID.
        """
        return MalURLClient(cid, net, client_dataloaders[int(cid)]['train'], client_dataloaders[int(cid)]['validation'])

    return client_fn


def main(args):
    net = init_model(num_labels=NUM_LABELS, fine_tune=True)

    input_dataset = load_raw_data(args.i)

    if args.split == 'even':
        num_clients = 10
        client_datasets, server_test_dataset = prepare_train_test_iid(input_dataset, num_clients)
    elif args.split == 'bias':
        num_clients = 3
        client_datasets, server_test_dataset = prepare_train_test_noniid(input_dataset, num_clients)
    else:
        raise Exception(f"Input split {args.split} not supported")

    client_dataloaders, server_testloader = process_data(client_datasets, server_test_dataset)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit = 2/num_clients, # Sample 2 available clients for training in each epoch
        evaluate_metrics_aggregation_fn = weighted_average, # Use weighted average function to aggregate the local evaluation metrics of clients. 
        fraction_evaluate = 2/num_clients, # Sample 2 available clients for model evaluation
        evaluate_fn=get_evaluate_fn(net, server_testloader)  # Pass the evaluation function
    )

    client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(client_dataloaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.epoch),
        strategy=strategy, # Server side strategy discussed in Section Server
        client_resources=client_resources,
    )


if __name__ == '__main__':
    main()