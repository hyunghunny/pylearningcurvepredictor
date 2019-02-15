import unittest
import numpy as np
import os
import argparse

from subprocess import Popen, PIPE

#from caffe.proto import caffe_pb2
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


def run_program(cmds):
    process = Popen(cmds, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    return exit_code


class ETRValidator(unittest.TestCase):
    def test_conservative_real_example(self):
        """
            The termination criterion expects the learning_curve in a file
            called learning_curve.txt as well as the current best value in 
            ybest.txt. We create both files and see if the termination criterion
            correctly predicts to cancel or continue running under various artificial
            ybest.
        """
        # sample learning curve selected from data2
        lr = [0.903699995,0.953399998,0.959299999,0.963500001,0.967099999,0.968000003,0.970500002,0.972599998,0.9733,0.973599998,0.976699999,0.9793,0.975799997,0.9789,0.976200002]
        lr_p = lr[:7]
        ybest = max(lr)
        xlim = 70
        prob_types = ["posterior_mean_prob_x_greater_than"]#["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]
        for prob_x_greater_type in prob_types:
            np.savetxt("learning_curve.txt", lr_p)
            write_xlim(xlim)

            open("ybest.txt", "w").write(str(ybest))
            open("termination_criterion_running", "w").write("running")

            ret = main(mode="conservative",
                prob_x_greater_type=prob_x_greater_type,
                nthreads=4)
            #ybest is higher than what the curve will ever reach
            #hence we expect to cancel the run:
            self.assertEqual(ret, 0)

            self.assertTrue(os.path.exists("y_predict.txt"))
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

    def test_predict_no_cancel(self):
        pass



if __name__ == "__main__":
    unittest.main()