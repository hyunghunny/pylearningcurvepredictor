import os
import json
import argparse
import time

import numpy as np

from pylrpredictor.curvefunctions import  all_models, model_defaults
from pylrpredictor.terminationcriterion import main

import ws.shared.lookup as lookup

class LearningCurveReader(object):
    def __init__(self, surrogate, **kwargs):
        self.name = "LR_{}".format(surrogate)
        l = lookup.load(surrogate)
        if l == None:
            raise TypeError("Fail to load surrogate")
        self.lrs = l.get_accuracies_per_epoch()
    
    def get_lr(self, index):
        if index < 0 or index >= len(self.lrs):
            raise ValueError('Invalid index: {}'.format(index))
        lr = self.lrs.ix[index].tolist()
        return lr

    def count(self):
        return len(self.lrs)


class LearningCurvePredictorEvaluator(object):
    
    def __init__(self, lcr, ybest=1.5, cp_ratio=0.5):
        self.lcr = lcr # learning curve reader
        lr = self.lcr.get_lr(0)
        self.xlim = len(lr)
        self.ybest = ybest
        self.num_checkpoint = int(self.xlim * 0.5)

        self.modes = ["conservative", "optimistic"]
        self.prob_types = ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]
        self.results = {}

    def add_result(self, index, result, save=True):
        self.results[str(index)] = result

        if save:
            save_file = "{}.json".format(self.lcr.name)
            with open(save_file, "w") as out:
                json.dump(self.results, out)

    def write_xlim(self, test_interval=1):
        #solver = caffe_pb2.SolverParameter()
        solver = {}
        
        #solver.max_iter = xlim * test_interval
        #solver.test_interval = test_interval
        solver['max_iter'] = self.xlim * test_interval
        solver['test_interval'] = test_interval

        with open("caffenet_solver.prototxt", "w") as jsonfile:
            json.dump(solver, jsonfile)
        pass

    def cleanup(self):
        if os.path.exists("learning_curve.txt"):
            os.remove("learning_curve.txt")
        if os.path.exists("ybest.txt"):    
            os.remove("ybest.txt")
        if os.path.exists("termination_criterion_running"):    
            os.remove("termination_criterion_running")
        if os.path.exists("term_crit_error.txt"):    
            os.remove("term_crit_error.txt")

    def prepare(self, lr, num_checkpoint):
        lr_p = lr[:num_checkpoint]
        np.savetxt("learning_curve.txt", lr_p)
        self.write_xlim()
        open("ybest.txt", "w").write(str(self.ybest))
        open("termination_criterion_running", "w").write("running")

    def run(self, index, modes=None, prob_types=None, num_checkpoint=None):
        if modes == None:
            modes = self.modes
        if prob_types == None:
            prob_types = self.prob_types
        
        lr = self.lcr.get_lr(index)
        if num_checkpoint == None:
            num_checkpoint = int(self.xlim * 0.5)
        result = { 
            "lr" : lr,
            "checkpoint" : self.num_checkpoint,
            "max_acc" : max(lr),
            "y_best" : self.ybest
             }
        for mode in modes:
            for prob_type in prob_types:
                key = "{}-{}".format(mode, prob_type)
                self.prepare(lr, num_checkpoint)
                ret = main(mode=mode,
                    prob_x_greater_type=prob_type,
                    nthreads=4)
                print("{}:{}-{}:{} returns {}".format(
                    self.lcr.name, index,
                    mode, prob_type, ret))
                y_predict = None
                if os.path.exists("y_predict.txt"):
                    y_predict = float(open("y_predict.txt").read())
                result[key] = {
                    "y_predict" : y_predict,
                    "returns" : ret}

                self.cleanup()
        
        self.add_result(index, result)


def single_test(surrogate, index):
    lcr = LearningCurveReader(surrogate)
    lcpe = LearningCurvePredictorEvaluator(lcr)
    lcpe.run(index)


def evaluate(surrogate):    
    lcr = LearningCurveReader(surrogate)
    lcpe = LearningCurvePredictorEvaluator(lcr)
    for i in range(lcr.count()):
        try:
            lcpe.run(i)
        except Exception as ex:
            print("Exception occurs at {}: {}".format(i, ex))
            continue    

if __name__ == "__main__":
    start_time = time.time()
#    single_test("data207", 5254)
    parser = argparse.ArgumentParser(description='Learning curve predictor evaluation.')
    parser.add_argument('surrogate', type=str, help='surrogate benchmark')

    args = parser.parse_args()
    evaluate(args.surrogate)
    print("It takes {} secs".format(int(time.time() - start_time)))
