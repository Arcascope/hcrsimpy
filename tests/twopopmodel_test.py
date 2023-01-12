import unittest

import numpy as np
from hcrsimpy.models import TwoPopulationModel


class TwoPopModelTests(unittest.TestCase):
            
    def test_darkness_integration(self):
        model = TwoPopulationModel()
        ts = np.arange(0,24*30, 0.10)
        light_est = np.zeros(len(ts))
        sol = model.integrate_model(ts, light_est, 
                                    np.array([1.0, 1.0, 0.0,0.0, 0.0]))
        self.assertAlmostEquals(186.8826, 
                                sol[2,-1],
                                places=4,
                                msg = "Darkness integration not running with the correct period")
        
    def test_darkness_period(self):
        model = TwoPopulationModel()
        ts = np.arange(0,24*30, 0.10)
        light_est = np.zeros(len(ts))
        u0 = np.array([1.0, 1.0, 0.0,0.0, 0.0])
        
        dlmo = model.integrate_observer(ts, light_est, u0) 
        self.assertAlmostEqual(np.diff(dlmo)[-1], 24.2, msg = "Darkness period is incorrect")
        
        cbt = model.integrate_observer(ts, light_est, u0, observer=TwoPopulationModel.CBTObs) 
        self.assertAlmostEqual(np.diff(cbt)[-1], 24.2, msg = "Darkness period is incorrect")

if __name__ == '__main__':
    unittest.main()
