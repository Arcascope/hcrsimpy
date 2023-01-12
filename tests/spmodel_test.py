import unittest

import numpy as np
from hcrsimpy.models import SinglePopModel


class SinglePopModelTests(unittest.TestCase):
            
    def test_darkness_integration(self):
        model = SinglePopModel()
        ts = np.arange(0,24*30, 0.10)
        light_est = np.zeros(len(ts))
        sol = model.integrate_model(ts, light_est, np.array([1.0, 0.0,0.0]))
        self.assertAlmostEquals(187.05691072, 
                                sol[1,-1],
                                msg = "Darkness integration not running with the correct period")


if __name__ == '__main__':
    unittest.main()
