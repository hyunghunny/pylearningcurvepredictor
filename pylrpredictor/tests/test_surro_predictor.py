import unittest
import numpy as np
import os
import json

from pylrpredictor.curve_predicter import SurrogateLearningCurvePredictor

class SurrogatePredictorTest(unittest.TestCase):

    def test_load_data207(self):
        pred = SurrogateLearningCurvePredictor("data207")

        cp = pred.get_preset_checkpoint()
        self.assertEqual(cp, 50)

    def test_prediction_time(self):
        pred = SurrogateLearningCurvePredictor("data207")

        t = pred.get_prediction_time(0)
        self.assertGreater(t, 0)

    def test_termination_true(self):
        pred = SurrogateLearningCurvePredictor("data207")
        for i in range(2):
            curr = pred.get_best_acc(i)
            pred.set_current_best(curr)
        term = pred.check_termination(6)
        self.assertEqual(term, True)

    def test_termination_false(self):
        pred = SurrogateLearningCurvePredictor("data207")
        for i in range(2):
            curr = pred.get_best_acc(i)
            pred.set_current_best(curr)
        term = pred.check_termination(3)
        self.assertEqual(term, True)
