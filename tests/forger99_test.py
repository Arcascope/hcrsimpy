import unittest

import numpy as np
from hcrsimpy.models import Forger99Model


class Forger99ModelTests(unittest.TestCase):
    def test_model_parameters(self):
        model = Forger99Model()
        self.assertEqual(model.delta, 0.0075)
    
    def test_model_alter_parameters(self):
        model = Forger99Model()
        model.set_parameters({"taux": 12.1, "kparam": 0.9})
        expected_return_dict = {
            "taux": 12.1,
            "mu": 0.23,
            "G": 33.75,
            "alpha_0": 0.05,
            "delta": 0.0075,
            "p": 0.50,
            "I0": 9500.0,
            "kparam": 0.9
        }
        self.assertDictEqual(model.get_parameters(), expected_return_dict)
    
    # def test_model_update_invalid_values(self):
    #     model = Forger99Model()
    #     # Invalid_Values not defined; error with assertRaises()
    #     # arguments 
    #     self.assertRaises(ValueError, model.updateParameters(),
    #                       {"taux": "abc"})
    
    def test_model_alpha_0(self):
        model = Forger99Model()
        #    expected_return_val = 0
        self.assertNotEqual(model.alpha0(8), -10000)
        self.assertEqual(model.alpha0(0), 0)
