import unittest
import sys
sys.path.append('../classes')
import myclass as mc

mytask = mc.Task('Game_1','Football')

class TestMyCode(unittest.TestCase):

    def test_adefault(self):
        self.assertEqual(mytask.default_time,10)

    def test_bincrease(self):
        mytask.increase_time()
        self.assertEqual(mytask.default_time, 11)
        mytask.increase_time()
        self.assertEqual(mytask.default_time, 12)
    
    def test_cdecrease(self):
        mytask.decrease_time()
        self.assertEqual(mytask.default_time, 11)
        mytask.decrease_time()
        self.assertEqual(mytask.default_time, 10)
        
    def test_dreset(self):
        mytask.reset_time()
        self.assertEqual(mytask.default_time, 0)

if __name__ == '__main__':
    unittest.main()