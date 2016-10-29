pylearningcurvepredictor
========================

predicting learning curves in python


Paper
------
Tobias Domhan, Jost Tobias Springenberg, Frank Hutter. Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves. IJCAI, 2015.

Requirements
------------
- numpy >= 1.7
- emcee>=2.1.0
- scipy>=0.13.3
- docutils>=0.3
- setuptools
- matplotlib
- triangle

Installation
------------
To install the learning curve predictor you can clone the repository and install it manually via

.. code-block:: shell
	python setup.py install

Basic Usage (Standalone)
-----------------------
An example how to use the curve predictor on artificial data. The same procedure applies for any other curve model and function.

.. code-block:: python
	from pylrpredictor.curvefunctions import  all_models, model_defaults
	from pylrpredictor.curvemodels import MCMCCurveModel, MLCurveModel
	import numpy as np


	#generate some data for the model
	x = np.arange(1, 1000)
	MCMC_model = curvemodels.MCMCCurveModel(function=all_models['log_power'],
                                        default_vals=model_defaults['log_power'],
                                        function_der=None)

	params = MCMC_model.default_function_param_array()
	params =  params + np.random.rand(params.shape[0])
	y = MCMC_model.function(x, *params)
	std = 0.01
	y += std*np.random.randn(y.shape[0])
	MCMC_model.fit(x,y)
	pred = MCMC_model.predict(np.array(x))

1. Specify which kind of curve model you want to use
2. Simply fit and predict the model with your x and y. 
3. You can also plot your results:

.. code-block:: python
	import mcmcmodelplotter

	plotter = mcmcmodelplotter.MCMCCurveModelPlotter(MCMC_model)
	plotter.predictive_density_plot(x)

License
-------
TBD
