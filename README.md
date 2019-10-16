# [DLND] Project: Predicting Bike-Sharing Patterns

Neural Network developed during my Deep Learning Fundamentals Nanodegree at Udacity 2018.
In this project, I created a simple neural network to use it to predict daily bike rental ridership.

## Getting Started

This Bike-Sharing-Dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the cnt column. You can see the first few rows of the data above.

Below is a plot showing the number of bike riders over the first 10 days or so in the data set. (Some days don't have exactly 24 entries in the data set, so it's not exactly 10 days.) You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model.

![dataset](./assets/dataset.png)

### Prerequisites

Thinks you have to install or installed on your working machine:

* Python 3.7
* Numpy (win-64 v1.15.4)
* Pandas (win-64 v0.23.4)
* Matplotlib (win-64 v3.0.2)
* Jupyter Notebook
* Torchvision (win-64 v0.2.1)
* PyTorch (win-64 v0.4.1)

### Environment:
* [Miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)

### Installing

Use the package manager [pip](https://pip.pypa.io/en/stable/) or
[miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/) to install your packages.  
A step by step guide to install the all necessary components in Anaconda for a Windows-64 System:
```bash
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
pip install torchvision
conda install -c pytorch pytorch
```

## Jupyter Notebook
* `Your_first_neural_network.ipynb`

This jupyter notebook describe the whole project from udacity, from the beginning to the end.

## Running the project

The whole project is located in the jupyter notebook file `Your_first_neural_network.ipynb` and it's include the training an the prediction part. The neural network is implemented in the file `my_answer.py` and used from the jupyter notebook.

### Parameters of training

To change to interations, the amount of hidden notes and some other parameters for the neural network, you can adapt these global constants inside the python file `my_answer.py` and watch the results.

```python
#########################################################
# Set your hyperparameters here
##########################################################
iterations      = 5000      # 5000 / 10000 / 100000 / 200000
learning_rate   = 0.6       # 0.6     
hidden_nodes    = 20        # 20
output_nodes    = 1         # 1
```
I got my best results with these values of parameters.

## Checkout the training results

```bash
Progress: 100.0% ... Training loss: 0.055 ... Validation loss: 0.139
```
See the visual results over iteration:

![training results](./assets/result_training.png)

## Checkout the prediction results

I use the test data to show how well the neural network is modeling the data.

![training results](./assets/result_prediction.png)

## Improvements

This is actually my best version of bike-sharing-patterns.
The next steps will be:
* do some experiments with the neural network to change the hyperparameters / hidden units / optimizer / structure of fully connected layers, ...
* convert the image classifier algorithm to keras
* convert the image classifier algorithm to fast.ai
* I will see what's comming next ...


## License
[MIT](https://choosealicense.com/licenses/mit/)
