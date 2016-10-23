from pydnn import preprocess
import unittest
import numpy as np

class TestTrainingData(unittest.TestCase):
    def test_split_even(self):
        data = np.array([np.arange(512), np.arange(512)])
        sets = preprocess.split_training_data(data, 64, 50, 25, 25)
        self.assertEqual(len(sets), 3)
        self.assertEqual(len(sets[0][0]), 256)
        self.assertEqual(len(sets[1][0]), 128)
        self.assertEqual(len(sets[2][0]), 128)

    def test_split_much_waste(self):
        data = np.array([np.arange(100), np.arange(100)])
        sets = preprocess.split_training_data(data, 17, 52, 24, 24)
        self.assertEqual(len(sets), 3)
        self.assertEqual(len(sets[0][0]), 51)
        self.assertEqual(len(sets[1][0]), 17)
        self.assertEqual(len(sets[2][0]), 17)

    def test_split_extra_batch(self):
        data = np.array([np.arange(100), np.arange(100)])
        sets = preprocess.split_training_data(data, 17, 50, 25, 25)
        self.assertEqual(len(sets), 3)
        self.assertEqual(len(sets[0][0]), 51)
        self.assertEqual(len(sets[1][0]), 17)
        self.assertEqual(len(sets[2][0]), 17)

    def test_dummy(self):
        data = np.array([np.arange(100), np.arange(100)])
        sets = preprocess.split_training_data(data, 17, 100, 0, 0)
        self.assertEqual(len(sets), 3)
        self.assertEqual(len(sets[0][0]), 85)
        self.assertEqual(len(sets[1][0]), 17)
        self.assertEqual(len(sets[2][0]), 17)