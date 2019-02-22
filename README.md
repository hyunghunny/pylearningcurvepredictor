# Performance evaluation of pylearningcurvepredictor
Evaluating the predicting learning curves in python


## Paper
Tobias Domhan, Jost Tobias Springenberg, Frank Hutter. Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves. IJCAI, 2015.

## Installation

If you are starting from a barebone environment, Install folllowing basic packages:
```
conda install numpy scipy matplotlib
``` 

A few more packages from mab/hpo environment are also required as follows:
```
pip install emcee
conda install -c conda-forge lmfit
```

If you are runing on Windows, Install the following package: triangle
```
python -m pip install packages\triangle-20190115.1-cp27-cp27m-win_amd64.whl
```

Otherwise, install it from pip as follow:
```
pip install triangle
```

## Usage

Run as follows to evaluate learning curve prediction performance with surrogate learning curves:
```
(prompt) python main.py {surrogate name}
```

Following names are the surrogates which are available:
 * data2
 * data3
 * data10
 * data20
 * data30
 * data207

