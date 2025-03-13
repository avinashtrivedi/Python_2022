import unittest
import sys
sys.path.append('../functions')
import print_date,validate

class TestMyCode(unittest.TestCase):

    def test_print(self):
        self.assertEqual(print_date.print_date('2','12','2020'), '2/12/2020')

    def test_validate(self):
        self.assertEqual(validate.validate('2','12','2020'), True)
        self.assertEqual(validate.validate('2','30','2020'), False)

if __name__ == '__main__':
    unittest.main()