import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper1(self):
        print('test_upper1')
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper1(self):
        print('test_upper2')
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split1(self):
        print('test_upper3')
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()