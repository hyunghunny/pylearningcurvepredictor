import unittest
import numpy as np
import os
import argparse
import json

import ws.shared.lookup as lookup

class SurrogateLoaderTest(unittest.TestCase):

    def test_load_data2(self):
        surrogate = 'data2'
        l = lookup.load(surrogate)
        lrs = l.get_accuracies_per_epoch()
        self.assertEqual(20000, len(lrs))
        lr = lrs.ix[0].tolist()
        self.assertEqual(15, len(lr))