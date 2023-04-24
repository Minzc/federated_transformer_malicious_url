import torch
import flwr as fl
from typing import List, Dict
from util import weighted_average, get_evaluate_fn, init_model, NUM_LABELS, load_raw_data, prepare_train_test_iid, prepare_train_test_noniid, process_data
from client import MalURLClient
import torch
from typing import List, Dict, Any, Callable
from parse_args import args
import json

def get_client_fn(client_dataloaders: List[Dict[str, Any]], net: torch.nn.Module, epoch: int) -> Callable[[str], Any]:
    """
    Return the function to create a client.

    Parameters
    ----------
    client_dataloaders : List[Dict[str, Any]]
        A list of dictionaries containing the data loaders for the training and validation data of all clients.
    net : torch.nn.Module
        The neural network to use for the clients.
    epoch : int
        The number of local training epoch
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
        return MalURLClient(cid, net, client_dataloaders[int(cid)]['train'], client_dataloaders[int(cid)]['validation'], epoch)

    return client_fn


def main():
    net = init_model(num_labels=NUM_LABELS, fine_tune=True)

    input_dataset = load_raw_data(args.input)

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

    client_resources = {"num_gpus": 0}

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(client_dataloaders, net, args.client_epoch),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.epoch),
        strategy=strategy, # Server side strategy discussed in Section Server
        client_resources=client_resources,
    )
    
    rst = {
        'metrics_distributed': history.metrics_distributed,
        'metrics_centralized': history.metrics_centralized,
    }
    with open(args.output, "w") as w:
        w.write(f"{json.dumps(rst)}\n")

if __name__ == '__main__':
    main()