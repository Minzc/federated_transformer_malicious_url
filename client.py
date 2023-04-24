import torch
from torch.utils.data import DataLoader
import flwr as fl
from typing import List
from util import test, train, set_parameters
import numpy as np

    
class MalURLClient(fl.client.NumPyClient):
    def __init__(self, cid: str, 
                 net: torch.nn.Module, 
                 trainloader: DataLoader, 
                 valloader: DataLoader,
                 epoch: int) -> None:
        """
        Initializes the class with the specified parameters.

        Parameters
        ----------
        cid : str
            A string representing the ID of the class.
        net : torch.nn.Module
            The neural network to use in the class.
        trainloader : DataLoader
            The data loader for the training set.
        valloader : DataLoader
            The data loader for the validation set.
        epoch : int
            The number of epochs to train for.

        Returns
        -------
        None
        """
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.epoch = epoch


    def get_parameters(self, config: dict) -> List[np.ndarray]:
        """
        Returns a list of the parameters of the neural network in the class.

        Parameters
        ----------
        config : dict
            A dictionary containing configuration parameters.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays containing the parameters of the neural network.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


    def fit(self, parameters, config):
        """
        Parameters
        ----------
        parameters: 
            The model parameters received from the central server.
        config: 
            Configuration parameters which allow the
            server to influence training on the client. It can be used to communicate arbitrary values from the server to the client,
            for example, to set the number of (local) training epochs.
        Returns
        -------
        parameters: 
            The locally updated model parameters.
        num_examples:
            The number of examples used for training.
        metrics:
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        set_parameters(self.net, parameters)
        print("Training Started...")
        # train_client function train the local model using the client' local dataset.
        # train_client function is defined in the utility file
        train(self.net, self.trainloader, epochs=self.epoch)
        print("Training Finished.")
        return self.get_parameters(config), len(self.trainloader), {}


    def evaluate(self, parameters, config):
        """
        Evaluate the provided parameters using the locally held dataset.
        Parameters
        ----------
        parameters :
            The current (global) model parameters.
        config : 
            Same as the config in fit function.
        Returns
        -------
        loss : 
            The evaluation loss of the model on the local dataset.
        num_examples : 
            The number of examples used for evaluation.
        metrics : 
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.
        """
        self.net = set_parameters(self.net, parameters)
        # test function is defined in the utility file.
        valid_loss, valid_accuracy, valid_f1 = test(self.net, self.valloader)
        metrics = {
            "valid_accuracy": float(valid_accuracy), 
            "valid_loss": float(valid_loss),
            'valid_f1': float(valid_f1),
        }
        return float(valid_loss), len(self.valloader), metrics