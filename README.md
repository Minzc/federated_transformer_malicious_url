# Federated Malicious URL Detection with Flower and Transformers

This is an introductory example to use Flower and Transformers to build a federated malicious URL classifier. For more details please refer to the blog report [[LINK]](https://minzc.github.io/posts/fed_trans/)

## Project Steup

Start by cloning the example project

This will create a new direcotyr called `federated_transformer_malicious_url` containing the following files:

```
-- client.py
-- parse_args.py
-- server.py
-- util.py
```

Project dependencies are defined in `requirements.txt`. It is recommended to use [Conda](https://docs.conda.io/en/latest/) to install those dependencies. 

## Run Federated Leraning with Flower and Transformers
Afterwards you are ready to start the simulation of federated learning process. You can simply run the command in a terminal as followings:

```
python server.py -e 10 -i malicious_phish.csv -s even -c 1 -o testoutput.json
```

```
Parameter Help
-e [Number of epoches in federated learning]
-i [Path to input file]
-s [How to split the input data to clients]
-c [Number of training epoches in clients' local training]
-o [Path to output file, which stores in the federated and central evaluation results in the process of federated learning]
```
