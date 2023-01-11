import unittest

import numpy as np
from hcrsimpy.models import SinglePopModel


class SinglePopModelTests(unittest.TestCase):
    
    def testAlpha(self):
        sp = SinglePopModel(lambda t: 100.0)
        for t in np.arange(0, 100, 0.1):
            aval = sp.alpha0(t)
            self.assertGreaterEqual(aval, 0.0, msg="Test alpha function")


if __name__ == '__main__':
    unittest.main()
