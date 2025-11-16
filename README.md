# <center>Split Fedrated Learning base on BERT Code Execution Guide
## 1.Python Environment
python version: 3.10

requirements: matplotlib, numpy, torch, collections, pickle, tqdm, struct, hashlib, datasets, sklearn, seaborn, transformers

## 2.Document Description
- aggregate.py: use different method(average/elited) to aggregrate models after training finished
- calculate.py: calculate various parameters in model training
- client_v7.py and server_v7.py: main training code
- client_v7_single.py: train model only on client(speed up, skip communication with server), for getting result faster
- config.py: all seetings about training
- connect.py: communication for client and server, CSE encryption
- dataset.py: load and preprocess dataset
- evaluate.py: evaluate final model on test dataset
- original_bert_train.py: train model with no federated method
- split_model.py: model split method

## 3.Running Commend

### 3.1 pull code:
``` cmd
git clone https://github.com/lucky-minion/DSA4213-finalproject.git
cd DSA4213-finalproject/v5(ESE)
```

### 3.2 set training parameters on config.py
- set model/dataset path
- write the right ip/port of client/server 
- set split layer and last layer of model
- set learning rate, batch size, epoch num, aggregate round, elite aggregate weight, client num

### 3.3 start training
start server first:
```
python server_v7.py
```
then run client:
```
python client_v7.py
```

### 3.4 aggregate federated trained model
run this commend:
```
python aggregate.py
```

### 3.5 train on traditional method
run commend:
```
python original_bert_train.py
```

### 3.6 test model performance:
set chosed modelpart0, modelpart1 and modelpart2 path for different training methods. then run the commend repeatedly:
```
python evaluate.py

```
