import unittest
import numpy as np
import os
import json

from pylrpredictor.curvefunctions import  all_models, model_defaults
from pylrpredictor.terminationcriterion import main


def write_xlim(xlim, test_interval=1):
    #solver = caffe_pb2.SolverParameter()
    solver = {}
    
    #solver.max_iter = xlim * test_interval
    #solver.test_interval = test_interval
    solver['max_iter'] = xlim * test_interval
    solver['test_interval'] = test_interval
    
    #open("caffenet_solver.prototxt", "w").write(str(solver))
    with open("caffenet_solver.prototxt", "w") as jsonfile:
        json.dump(solver, jsonfile)
    pass

# sample learning curve selected from data2
lr = [0.903699995,0.953399998,0.959299999,0.963500001,0.967099999,0.968000003,0.970500002,0.972599998,0.9733,0.973599998,0.976699999,0.9793,0.975799997,0.9789,0.976200002]
xlim = len(lr)
num_checkpoint = int(xlim * 0.5)
lr_p = lr[:num_checkpoint]
ybest = 1.5


results = {}

class ETRValidationTest(unittest.TestCase):

    def update_result(self, mode, prob_x_greater_type, lr, num_checkpoint, y_predict):
        key = "{}_{}".format(mode, prob_x_greater_type)
        result = {
        "lr": lr,
        "num_checkpoint" : num_checkpoint,
        "y_predict" : y_predict,
        "y_actual": max(lr)        
        }
        results[key] = result

        with open("results.json", "w") as out:
            json.dump(results, out)
        
    def test_data2_optimistic_mean_prob_example(self):
        
        prob_types = ["posterior_mean_prob_x_greater_than"]
        for mode in ["optimistic"]:
            for prob_x_greater_type in prob_types:
                np.savetxt("learning_curve.txt", lr_p)
                write_xlim(xlim)

                open("ybest.txt", "w").write(str(ybest))
                open("termination_criterion_running", "w").write("running")

                ret = main(mode=mode,
                    prob_x_greater_type=prob_x_greater_type,
                    nthreads=4)
               
                self.assertTrue(os.path.exists("y_predict.txt"))                
                y_predict = float(open("y_predict.txt").read())
                print("{} predicted accuracy: {}".format(prob_x_greater_type, y_predict))
                self.update_result(mode, prob_x_greater_type, lr, num_checkpoint, y_predict)
                self.assertFalse(os.path.exists("termination_criterion_running"))
                self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()
    
    def test_data2_optimistic_prob_example(self):
        prob_types = ["posterior_prob_x_greater_than"]
        for mode in ["optimistic"]:
            for prob_x_greater_type in prob_types:
                np.savetxt("learning_curve.txt", lr_p)
                write_xlim(xlim)

                open("ybest.txt", "w").write(str(ybest))
                open("termination_criterion_running", "w").write("running")

                ret = main(mode=mode,
                    prob_x_greater_type=prob_x_greater_type,
                    nthreads=4)
                
                self.assertTrue(os.path.exists("y_predict.txt"))                
                y_predict = float(open("y_predict.txt").read())
                print("{} predicted accuracy: {}".format(prob_x_greater_type, y_predict))
                self.update_result(mode, prob_x_greater_type, lr, num_checkpoint, y_predict)
                self.assertFalse(os.path.exists("termination_criterion_running"))
                self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()

    def test_data2_conservative_mean_prob_example(self):
        prob_types = ["posterior_mean_prob_x_greater_than"]
        for mode in ["conservative"]:
            for prob_x_greater_type in prob_types:
                np.savetxt("learning_curve.txt", lr_p)
                write_xlim(xlim)

                open("ybest.txt", "w").write(str(ybest))
                open("termination_criterion_running", "w").write("running")

                ret = main(mode=mode,
                    prob_x_greater_type=prob_x_greater_type,
                    nthreads=4)
                
                self.assertTrue(os.path.exists("y_predict.txt"))                
                y_predict = float(open("y_predict.txt").read())
                print("{} predicted accuracy: {}".format(prob_x_greater_type, y_predict))
                self.update_result(mode, prob_x_greater_type, lr, num_checkpoint, y_predict)
                self.assertFalse(os.path.exists("termination_criterion_running"))
                self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()

    def test_data2_conservative_prob_example(self):

        prob_types = ["posterior_prob_x_greater_than"]
        for mode in ["conservative"]:
            for prob_x_greater_type in prob_types:
                np.savetxt("learning_curve.txt", lr_p)
                write_xlim(xlim)

                open("ybest.txt", "w").write(str(ybest))
                open("termination_criterion_running", "w").write("running")

                ret = main(mode=mode,
                    prob_x_greater_type=prob_x_greater_type,
                    nthreads=4)
                
                self.assertTrue(os.path.exists("y_predict.txt"))                
                y_predict = float(open("y_predict.txt").read())
                print("{} predicted accuracy: {}".format(prob_x_greater_type, y_predict))
                self.update_result(mode, prob_x_greater_type, lr, num_checkpoint, y_predict)
                self.assertFalse(os.path.exists("termination_criterion_running"))
                self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()

    def cleanup(self):
        if os.path.exists("learning_curve.txt"):
            os.remove("learning_curve.txt")
        if os.path.exists("ybest.txt"):    
            os.remove("ybest.txt")
        if os.path.exists("termination_criterion_running"):    
            os.remove("termination_criterion_running")
        if os.path.exists("term_crit_error.txt"):    
            os.remove("term_crit_error.txt")


if __name__ == "__main__":
    unittest.main()