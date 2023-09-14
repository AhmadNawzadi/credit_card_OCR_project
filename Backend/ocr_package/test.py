import unittest
from calc import get_number_length

class TestGetNumberLength(unittest.TestCase):
    def test_get_number_length_true(self):
        output = get_number_length(16)

        self.assertEqual(output, True)
