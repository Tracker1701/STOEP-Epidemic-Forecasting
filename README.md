

<h1>
    <span>STOEP: Spatio-Temporal priOr-aware Epidemic Predictor</span>
</h1>





This code is the official PyTorch implementation of STOEP  


## Data Description

#### jp20200401_20210921.npy 

contains a dictionary of three numpy array: 'node' for node features; 'SIR' for S, I, R data; 'od' for OD flow data.

#### Input and Output

* Input node features: historical daily confirmed cases, daily movement change, the ratio of daily confirmed cases in active cases and day of week. 
* Input for Case-aware Adjacency Learning  : OD flow data, Input node features
* Output: predicted daily confirmed cases

## Installation Dependencies

Working environment and major dependencies:

* Windows 11
* Python 3 (3.8; Anaconda Distribution)
* NumPy (1.26.4)
* Pytorch (2.9.0)

## Run Model

Download this project into your device, then run the following:

``
cd /model
``


Run the main program to train, validate and test on GPU 0:

``
python Main.py -GPU cuda:0
``
