# CS6910 Assignment 3
Name: Oikantik Nath | Roll: CS22S013 | Course: CS6910 Fundamentals of Deep Learning | [WandB Report](https://wandb.ai/dl_research/CS6910_Assignment3/reports/Assignment-3-CS6910---Vmlldzo0MDc1MTA4?accessToken=3yu9rwpccb5yfgb8fqroo0egi0y73jktso9vx7fz9e7kcho6e8wshgvnu229jnd7)
Task: Use recurrent neural networks to build a transliteration system.

## Question 1
Code can be accessed [here](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q1.ipynb). `keras.datasets` is used to get `fashion_mnist` dataset and 1 sample image is plotted for each class using `wandb.log()`.

## Questions 2-4
The neural network is implemented by the class `FeedForwardNN`, present in the `train.py` file.  Access it [here](https://github.com/oikn2018/CS6910_assignment_1/blob/main/train.py).

### Run the Code
To run the code, execute in cmd: 
#### Format:
`python train.py -wp <wandb_project_name> -we <wandb_entity_name> -e <epochs> -b <batch_size> -o <optimizer> -lr <learning_rate> -w_i <weight_initialization_method> -nhl <num_hidden_layers> -sz <size_hidden_layer> -a <activation_function>`

#### To test it on the best model achieved:
`python train.py -wp Testing -we dl_research -e 20 -b 64 -o nadam -lr 0.005 -w_i Xavier -nhl 5 -sz 512 -a sigmoid`

## Question 7
The confusion matrix code is available [here](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q7.py). It is tested on the Fashion-MNIST test data and the output confusion matrix is logged in the report.


## Question 10
Since the MNIST dataset is much simpler in terms of image complexity compared to Fashion-MNIST dataset which I have used in my experimentation, so I suggest the following 3 configurations that give me the best accuracy scores on the Fashion-MNIST dataset. You can access code [here](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py).

- Configuration 1: 
```python
config = { 
	"epochs" : 20,
	"learning_rate": 0.005,
	"no_hidden_layers": 5, 
	"hidden_layers_size": 512,
	"weight_decay": 0,
	"optimizer": "nadam",
	"batch_size": 64,
	"weight_initialization" : "xavier" ,
	"activations" : "sigmoid",
}
```
To run above configuration, download [code](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) from GitHub and execute following command on cmd:
`python Q10.py -wp Testing -we dl_research -d mnist -e 20 -b 64 -o nadam -lr 0.005 -w_i Xavier -nhl 5 -sz 512 -a sigmoid`

- Configuration 2: 
```python
config = { 
	"epochs" : 20,
	"learning_rate": 0.005,
	"no_hidden_layers": 5, 
	"hidden_layers_size": 256,
	"weight_decay": 0,
	"optimizer": "nadam",
	"batch_size": 32,
	"weight_initialization" : "xavier" ,
	"activations" : "tanh",
}
```
To run above configuration, download [code](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) from GitHub and execute following command on cmd:
`python Q10.py -wp Testing -we dl_research -d mnist -e 20 -b 32 -o nadam -lr 0.005 -w_i Xavier -nhl 5 -sz 256 -a tanh`


- Configuration 3: 
```python
config = { 
	"epochs" : 20,
	"learning_rate": 0.0001,
	"no_hidden_layers": 5, 
	"hidden_layers_size": 256,
	"weight_decay": 0,
	"optimizer": "adam",
	"batch_size": 128,
	"weight_initialization" : "xavier" ,
	"activations" : "relu",
}
```
To run above configuration, download [code](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) from GitHub and execute following command on cmd:
`python Q10.py -wp Testing -we dl_research -d mnist -e 20 -b 128 -o adam -lr 0.0001 -w_i Xavier -nhl 5 -sz 256 -a relu`

---
The codes are organized as follows:

| Question | Location | Function | 
|----------|----------|----------|
| Question 1 | [Question-1](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q1.ipynb) | Plotting Sample Images of Each Class | 
| Question 2-4 | [Question-2-4](https://github.com/oikn2018/CS6910_assignment_1/blob/main/train.py) | Feedforward Neural Network Training and Evaluating Accuracies |
| Question 7 | [Question-7](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q7.py) | Confusion Matrix for Test Data on Best Model | 
| Question 10 | [Question-10](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) | 3 Best Hyperparameter configurations for MNIST | 
