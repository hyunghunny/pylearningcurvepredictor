import numpy as np
import pandas as pd

def debug(args, **kwargs):
    print(args, kwargs)

class SurrogateLearningCurvePredictor(object):
    
    def __init__(self, preset, data_folder='./', **kwargs):
        self.method = 'function_mcmc_extrapolation'
        csv_path = data_folder + 'LCP_{}.csv'.format(preset)
        data = pd.read_csv(csv_path)

        self.checkpoints = data.ix[:, 1].values
        self.best_accs = data.ix[:, 2].values
        self.fatasies = data.ix[:, 3].values
        self.est_times = data.ix[:, 4].values
        
        self.current_best = None

    def get_preset_checkpoint(self, index=0):
        return int(self.checkpoints[index])

    def get_fatasy(self, index):
        return float(self.fatasies[index])

    def get_prediction_time(self, index):
        if np.isnan(self.est_times[index]) != True:
            return float(self.est_times[index])
        else:
            return float(np.nanmean(self.est_times)) 

    def get_best_acc(self, index):
        return float(self.best_accs[index])
    
    def set_current_best(self, best_acc):
        if self.current_best == None or self.current_best < best_acc: 
            self.current_best = best_acc

    def check_termination(self, index, current_epoch=None):
        if current_epoch != None:
            if self.get_preset_checkpoint(index) != current_epoch:
                debug("{} is not the checkpoint.".format(current_epoch))
                return False
        
        if self.current_best == None:
            return False
        elif self.get_fatasy(index) < self.current_best:
            return True
        else:
            return False
