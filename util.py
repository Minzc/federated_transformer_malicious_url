import random
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
import torch
from torch.utils.data import DataLoader
from typing import Callable, Tuple, Any

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_LINES = 65110 # Due to limited computation resources, we only load a subset of data points for our experiments.
BATCH_SIZE = 32
NUM_LABELS = 4
RANDOM_SEEDS = 42
CHECKPOINT = "distilbert-base-uncased"

def load_raw_data(path: str) -> Dataset:
    """
    This function loads the Malicious URL data, which includes information about URLs and their corresponding labels.

    Parameters
    ----------
    path: str
        File path

    Returns
    -------
    raw_datasets (Dataset):
        The raw dataset containing the URLs and their corresponding labels. 
        The dataset is shuffled and split into training and evaluation sets. 
        The features include the URL string and the corresponding label.
    """
    class_names = ['benign', 'malware', 'phishing', 'defacement']
    url_features = Features({'url': Value('string'), 'type': ClassLabel(names=class_names)})
    
    raw_datasets = load_dataset("csv", data_files=path, features=url_features, split=f"train[:{NUM_LINES}]")
    raw_datasets = raw_datasets.rename_column("type", "labels")
    raw_datasets = raw_datasets.shuffle(seed=RANDOM_SEEDS)
    return raw_datasets


def test(net: torch.nn.Module, testloader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
    """
    This function evaluates the performance of a given neural network on a test dataset using metrics such as accuracy and F1 score.
    
    Parameters
    -----------
    net: torch.nn.Module
        The neural network to be evaluated.
    testloader: torch.utils.data.DataLoader
        The dataloader for the test dataset.

    Returns
    -------
    loss: float
        The average loss of the network on the test dataset.
    accuracy: float
        The accuracy of the network on the test dataset.
    f1: float
        The F1 score of the network on the test dataset.
    """
    acc_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    loss = 0
    net.eval()
    with torch.no_grad():
      for batch in tqdm.tqdm(testloader):
          batch = {k: v.to(DEVICE) for k, v in batch.items()}
          with torch.no_grad():
              outputs = net(**batch)
          logits = outputs.logits
          loss += outputs.loss.item()
          predictions = torch.argmax(logits, dim=-1)
          acc_metric.add_batch(predictions=predictions, references=batch["labels"])
          f1_metric.add_batch(predictions=predictions, references=batch["labels"])
      loss /= len(testloader.dataset)
      accuracy = acc_metric.compute()["accuracy"]
      f1 = f1_metric.compute(average="macro")['f1']
    return loss, accuracy, f1


    
def _train_test_split(local_data: Dataset) -> DatasetDict:
    """
    This function splits a given local dataset of a client into training and validation datasets.

    Parameters
    ----------
    loca_data: Dataset
        The local dataset of a client to be split.

    Returns
    -------
    data_dict: DatasetDict
        The training and validation datasets of the client.
    """
    train_val_client_split = local_data.train_test_split(test_size=0.2, seed=RANDOM_SEEDS)  # 80% local training data, 20% local validation data
    data_dict = DatasetDict({
                'train': train_val_client_split['train'],
                'validation': train_val_client_split['test'],
                })
    return data_dict


def prepare_train_test_iid(raw_datasets: Dataset,  num_clients: int) -> Tuple[List[DatasetDict], Dataset]:
    """
    Prepares the training and testing datasets for a federated learning scenario where the data is partitioned across 
    multiple clients in an IID (Independent and Identically Distributed) manner.

    Parameters
    ----------
    raw_datasets: Dataset
        The raw dataset containing the URLs and their corresponding labels.

    Returns
    -------
    client_datasets: List[DatasetDict]
        A list of datasets for each client, each containing the training and validation subsets.
    server_test_dataset: Dataset
        The dataset used by the central server for central evaluation.
    """
    train_test_split = raw_datasets.train_test_split(test_size=0.2, seed=42)
    client_dataset = train_test_split['train']
    server_test_dataset = train_test_split['test']
    partition_size = len(client_dataset) // num_clients # `num_clients` is the total number of clients in the federated learning process. `partition_size` is the number of records in each client's local data.

    client_datasets = []
    for _ in range(num_clients):
        client_split = client_dataset.train_test_split(train_size=partition_size)
        client_dataset = client_split['test'] # The remaining data will be divided among the other clients.
        client_datasets.append(_train_test_split(client_split['train']))
    return client_datasets, server_test_dataset


def prepare_train_test_noniid(raw_datasets: datasets.Dataset, num_clients: int) -> Tuple[List[datasets.DatasetDict], datasets.Dataset]:
    """
    Prepares the training and testing datasets for a federated learning scenario where the data is partitioned across 
    multiple clients in a non-IID (Non-Independent and Identically Distributed) manner.

    Parameters
    ----------
    raw_datasets: datasets.Dataset
        The raw dataset containing the URLs and their corresponding labels.
    num_clients: int
        The total number of clients in the federated learning process.

    Returns
    -------
    client_datasets: List[datasets.DatasetDict]
        A list of datasets for each client, each containing the training and validation subsets.
    server_test_dataset: datasets.Dataset
        The dataset used by the central server for central evaluation.
    """

    train_test_split = raw_datasets.train_test_split(test_size=0.2, seed=42)
    clients_dataset, server_test_dataset = train_test_split['train'], train_test_split['test']
    # label_id 0: benign
    # label_id 1: malicious
    # label_id 2: phishing
    # label_id 3: defacement
    whole_benign = clients_dataset.filter(lambda x: x['labels'] == 0)
    benign_size_per_client = len(whole_benign) // num_clients

    client_datasets = []
    # Class 0 is benign
    for cid in range(num_clients):
        abnormal_urls = clients_dataset.filter(lambda x: x['labels'] == (cid + 1)) 
        client_split = whole_benign.train_test_split(train_size=benign_size_per_client, seed=42)
        local_benign, whole_benign = client_split['train'], client_split['test']

        local_dataset = concatenate_datasets([local_benign, abnormal_urls])
        client_datasets.append(_train_test_split(local_dataset))

    return client_datasets, server_test_dataset


def verfiy_data_loader(data_loader: torch.utils.data.DataLoader) -> None:
    """
    Verifies that a given data loader returns non-empty batches.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        The data loader instance to be verified.

    Returns
    -------
    None
    """
    for batch in data_loader:
        for value in batch.values():
            assert len(value) > 0, f'Size: {len(value)}'

# raw_datasets = load_raw_data()

# client_datasets, global_testset = prepare_train_test(raw_datasets)


def process_data(client_datasets: List[datasets.DatasetDict], global_testset: datasets.Dataset) -> Tuple[List[Dict[str, torch.utils.data.DataLoader]], torch.utils.data.DataLoader]:
    """
    Loads and tokenizes the training and testing datasets for each client.

    Parameters
    ----------
    client_datasets: List[datasets.DatasetDict]
        A list of datasets for each client, each containing the training and validation subsets.

    global_testset: datasets.Dataset
        The dataset used by the central server for central evaluation.

    Returns
    -------
    client_dataloaders: List[Dict[str, torch.utils.data.DataLoader]]
        A list of data loaders for each client, each containing the training and validation data.

    testloader: torch.utils.data.DataLoader
        The data loader instance for the testing data used by the central server.
    """

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(record):
        return tokenizer(record["url"], truncation=True, padding=True)

    client_dataloaders = []
    
    for client_dataset in client_datasets:
      tokenized_datasets = (client_dataset
                            .map(tokenize_function, batched=True)
                            .remove_columns("url"))

      # The main purpose of DataCollatorWithPadding is to dynamically pad input 
      # sequences in a batch with the padding token to match the longest sequence in that batch. 
      trainloader = DataLoader(
          tokenized_datasets["train"],
          shuffle=True,
          batch_size=BATCH_SIZE,
          collate_fn=data_collator,
      )

      valloader = DataLoader(
          tokenized_datasets["validation"], 
          batch_size=BATCH_SIZE, 
          collate_fn=data_collator
      )
      client_dataloaders.append({
          "train": trainloader,
          'validation': valloader
      })
      for data_loader in [trainloader, valloader]:
        verfiy_data_loader(data_loader)

    tokenized_test_datasets = (global_testset
                            .map(tokenize_function, batched=True)
                            .remove_columns("url"))
    testloader = DataLoader(
        tokenized_test_datasets, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator
    )

    verfiy_data_loader(testloader)

    return client_dataloaders, testloader
    
# client_dataloaders, testloader = load_data()

def init_model(num_labels: int, fine_tune: bool = True) -> torch.nn.Module:
    """
    Initialize a BERT based sequence classifier.

    Parameters
    ----------
    num_labels:
        The number of classes the model should predict.
    fine_tune:
        If we want to fine tune the parameters of the pre-trained BERT model.

    Returns
    -------
    net:
        A BERT-based sequence classifier model with the specified number of labels. 
    """
    net = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=num_labels).to(DEVICE)

    if fine_tune == False:
        for name, param in net.named_parameters():
            if name.startswith("bert"): # choose whatever you like here
                param.requires_grad = False

    net.train()
    return net


def train(net: nn.Module, trainloader: DataLoader, epochs: int) -> None:
    """
    Train the given neural network for the specified number of epochs using the given data loader.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The data loader containing the training data.
    epochs : int
        The number of epochs to train for.

    Returns
    -------
    None
    """

    optimizer = AdamW(net.parameters(), lr=5e-5)

    for _ in tqdm.tqdm(range(epochs), desc='epoch'):
      net.train()
      total_loss = 0
      for batch in tqdm.tqdm(trainloader, desc='iterate data'):
          batch = {k: v.to(DEVICE) for k, v in batch.items()}
          outputs = net(**batch)
          logits = outputs.get("logits")
          loss_fct = nn.CrossEntropyLoss(
                        weight=torch.tensor([1.0, 10.0, 10.0, 10.0], 
                        device=DEVICE)
                      )
          labels = batch.get("labels")
          loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          total_loss += loss.item()
      
    torch.cuda.empty_cache()


def set_parameters(net: nn.Module, parameters: List[torch.Tensor]) -> nn.Module:
    """
    Sets the parameters of a PyTorch neural network module to the specified tensors.

    Parameters
    ----------
    net: nn.Module
        The neural network module to set the parameters for.
    parameters: List[torch.Tensor]
        The list of tensors to set the parameters of the neural network module to.

    Returns
    -------
    nn.Module
        The neural network module with updated parameters.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_parameters(self, config):
    """
    Parameters
    ----------
    config:
        Configuration parameters requested by the server.
        This can be used to tell the client which parameters
        are needed along with some Scalar attributes.
    Returns
    -------
    parameters: 
        The local model parameters as a list of NumPy ndarrays.
    """
    return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Multiply accuracy of each client by number of examples used.
    Aggregate and return custom metric (weighted average).

    Parameters
    ----------
    metrics: List[Tuple[int, Metrics]]
        The list of local evaluation metrics sent by clients.
        metrics[idx] is the evaluation sent by the `idx` evaluation client.
        metrics[idx][0] is the number of records of the corresponding client.
        metrics[idx][1] is the evaluation metrics of the corresponding clients.

    Returns
    -------
    weight_metrics: Metrics
        The weighted average of the federated evaluation.
    """
    weight_metrics = {}
    for metric_name in metrics[0][1].keys():
      for num_examples, m in metrics:
        metric = [num_examples * m[metric_name] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        weight_metrics[metric_name] = sum(metric) / sum(examples)
    
    return weight_metrics


def get_evaluate_fn(net: torch.nn.Module, testloader: DataLoader) -> Callable[[int, Any, dict], Tuple[float, dict]]:
    """
    Return an evaluation function for centralized evaluation.

    Parameters
    ----------
    net : torch.nn.Module
        The model to be evaluated.
    testloader : DataLoader
        The dataset loader.

    Returns
    -------
    Callable[[int, Any, dict], Tuple[float, dict]]
        A function that evaluates the model on the given test data and returns the evaluation loss and metrics.
    """
    def evaluate(server_round: int, parameters: Any, config: dict) -> Tuple[float, dict]:
        """
        Evaluate the model on the given test data and return the evaluation loss and metrics.

        Parameters
        ----------
        server_round : int
            The current epoch of federated learning.
        parameters : Any
            The current (global) model parameters.
        config : dict
            Same as the config in fit.

        Returns
        -------
        Tuple[float, dict]
            A tuple containing the evaluation loss and a dictionary of evaluation metrics.
        """
        set_parameters(net, parameters)  # 'net' is the global model. Update model with the latest parameters
        loss, accuracy, f1 = test(net, testloader)
        return loss, {"accuracy": accuracy, 'f1': f1}

    return evaluate
