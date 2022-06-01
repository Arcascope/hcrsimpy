import unittest

from HCRSimPY.models import vdp_forger99_model
import numpy as np

class TestModel(unittest.TestCase):
    def test_model_parameters(self):
        model = vdp_forger99_model(lambda t: 0.0)
        self.assertEqual(model.delta, 0.0075)

    def test_model_alter_parameters(self):
        model = vdp_forger99_model(lambda t:0.0)
        model.updateParameters({"taux":12.1, "kparam":0.9})
        expected_return_dict={
            "taux":12.1,
            "mu":0.23,
            "G":33.75,
            "alpha_0":0.05,
            "delta":0.0075,
            "p":0.50,
            "I0":9500.0,
            "kparam":0.9
        }
        self.assertDictEqual(model.getParameters(), expected_return_dict)

    def test_model_update_invalid_values(self):
        model = vdp_forger99_model(lambda t:0.0)
        # Invalid_Values not defined; error with assertRaises()
        # arguments
        self.assertRaises(Invalid_Values,model.updateParameters,{"taux":"abc"})


    def test_model_alpha_0(self):
        model = vdp_forger99_model(lambda t:0.0)
        expected_return_val=0
        self.assertNotEqual(model.alpha0(8), -10000)
        self.assertEqual(model.alpha0(0),0)

    def test_derv_linear_function(self):
        model = vdp_forger99_model(lambda t: 3 * t)
        self.assertListEqual(model.derv(3,np.array([0,1,2])).tolist(), [3,3,3])
