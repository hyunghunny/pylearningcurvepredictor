import os
import json
import argparse
import time

import numpy as np
from subprocess import Popen, PIPE

from pylrpredictor.curvefunctions import  all_models, model_defaults
from pylrpredictor.terminationcriterion import main

import ws.shared.lookup as lookup

def run_program(cmds):
    process = Popen(cmds, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    return exit_code

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
    
    def __init__(self, lcr, cp_ratio=0.5):
        self.lcr = lcr # learning curve reader
        lr = self.lcr.get_lr(0)
        self.xlim = len(lr)
                
        self.num_checkpoint = int(self.xlim * cp_ratio)

        self.modes = ["conservative"] #, "optimistic"
        self.prob_types = ["posterior_prob_x_greater_than"] #"posterior_mean_prob_x_greater_than", 
        self.results = self.load_results()

    def load_results(self):
        save_file = "{}.json".format(self.lcr.name)
        if os.path.exists(save_file):
            with open(save_file) as json_file:
                return json.load(json_file)
        else:
            return {}

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
        #if os.path.exists("term_crit_error.txt"):    
        #    os.remove("term_crit_error.txt")

    def prepare(self, lr, num_checkpoint, ybest):
        lr_p = lr[:num_checkpoint]
        np.savetxt("learning_curve.txt", lr_p)
        self.write_xlim()
        open("ybest.txt", "w").write(str(ybest))
        open("termination_criterion_running", "w").write("running")

    def run(self, index, 
            modes=None, prob_types=None, 
            checkpoints=None, as_process=True, restore=True):
        if modes == None:
            modes = self.modes
        if prob_types == None:
            prob_types = self.prob_types
        
        lr = self.lcr.get_lr(index)
        if checkpoints == None:
            checkpoints = [self.num_checkpoint]
        r = { 
            "max_acc" : max(lr)
        }
        if str(index) in self.results:
            r = self.results[str(index)]
        
        start_time = None
        eval_time = None
        
        for num_checkpoint in checkpoints:
            for mode in modes:
                for prob_type in prob_types:
                    key = "{}-{}-{}".format(num_checkpoint, mode, prob_type)
                    ybest = max(lr) + 0.5 # Unrealistic fatasy to finding stopping prediction value. 
                    ret = 0
                    y_predict = None
                    if key in r and restore == True:
                        y_predict = r[key]['y_predict']
                        if 'y_best' in r[key]:
                            ybest = r[key]['y_best']
                        if y_predict != None:
                            print("Restore [{}] {}: {}".format(index, key, y_predict))
                    
                    if y_predict == None:
                        print("Run [{}] {}".format(index, key))
                        while ybest <= max(lr) + 0.5:
                            start_time = time.time()
                            self.prepare(lr, num_checkpoint, ybest)
                            if as_process == False:
                                ret = main(mode=mode, prob_x_greater_type=prob_type, nthreads=4)
                            else:
                                ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                                    "--nthreads", "5",
                                    "--mode", mode, 
                                    "--prob-x-greater-type", prob_type])                            
                            
                            print("{}:{}-{}-{}:{} returns {}".format(
                                self.lcr.name, index,
                                num_checkpoint, mode, prob_type, ret))
                            if ret == 1:
                                break
                            else:
                                #ybest += 0.5
                                # here means no termination.
                                break 
                    
                        if os.path.exists("y_predict.txt"):
                            y_predict = float(open("y_predict.txt").read())
                        else:
                            y_predict = 0.0
                    if start_time != None:
                        eval_time = time.time() - start_time
                    r[key] = {
                        "num_checkpoint" : num_checkpoint,
                        "y_predict" : y_predict,
                        "y_best" : ybest,
                        "est_time": eval_time 
                        }

                    self.cleanup()
        
        self.add_result(index, r)


def single_test(surrogate, index):
    lcr = LearningCurveReader(surrogate)
    lcpe = LearningCurvePredictorEvaluator(lcr)
    lcpe.run(index, restore=False)


def data207_default_test():
    lcr = LearningCurveReader('data207')
    lcpe = LearningCurvePredictorEvaluator(lcr)
    for index in range(lcr.count()):
        lcpe.run(index, checkpoints=[30, 60, 90])

def evaluate(surrogate, start_index, end_index=-1):    
    lcr = LearningCurveReader(surrogate)
    lcpe = LearningCurvePredictorEvaluator(lcr)
    if end_index == -1:
        end_index = lcr.count()
    print("Evaluating {} learning curves...".format(end_index - start_index))
    for i in range(start_index, end_index):
        try:
            print("Run with learning curve #{}...".format(i))
            lcpe.run(i)
        except Exception as ex:
            print("Exception occurs at {}: {}".format(i, ex))
            continue    

def run_all(start_time):
    parser = argparse.ArgumentParser(description='Learning curve predictor evaluation.')
    parser.add_argument('--start_index', type=int, default=0, help='Start index')
    parser.add_argument('--end_index', type=int, default=-1, help='End index, if -1, means all')
    parser.add_argument('surrogate', type=str, help='surrogate benchmark')

    args = parser.parse_args()
    evaluate(args.surrogate, args.start_index, args.end_index)
    

if __name__ == "__main__":
    start_time = time.time()
#    single_test("data3", 22)
    data207_default_test()
    #run_all(start_time)
    print("It takes {} secs".format(int(time.time() - start_time)))


