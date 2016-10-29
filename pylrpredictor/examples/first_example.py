import pylrpredictor.curvemodels
import numpy as np
from pylrpredictor.curvefunctions import  all_models, model_defaults
import pylrpredictor.mcmcmodelplotter


MLcurve_model = curvemodels.MLCurveModel(function=all_models['log_power'],
	                                 default_vals=model_defaults['log_power'],
	                                 recency_weighting=True)
#generate some data for the model
x = np.arange(1, 1000)
params = MLcurve_model.default_function_param_array()
params =  params + np.random.rand(params.shape[0])
y = MLcurve_model.function(x, *params)
std = 0.01
y += std*np.random.randn(y.shape[0])
print('starting with MLCurve model:')
MLcurve_model.fit(x, y)
print("original params vs fit params:")
print(params)
print(MLcurve_model.ml_params)
pred = MLcurve_model.predict(x)

print('switching to the MCMC model:')
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

plotter = mcmcmodelplotter.MCMCCurveModelPlotter(MCMC_model)
plotter.predictive_density_plot(x)