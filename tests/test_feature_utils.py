import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_utils import preprocess_landmarks


class FeatureUtilsTests(unittest.TestCase):
    def test_preprocess_landmarks_returns_63_features(self):
        sample = np.array([
            [0.10, 0.20, 0.00],
            [0.15, 0.18, 0.01],
            [0.20, 0.16, 0.02],
            [0.25, 0.14, 0.03],
            [0.30, 0.12, 0.04],
            [0.35, 0.10, 0.05],
            [0.40, 0.08, 0.06],
            [0.45, 0.07, 0.07],
            [0.50, 0.06, 0.08],
            [0.55, 0.05, 0.09],
            [0.60, 0.04, 0.10],
            [0.65, 0.03, 0.11],
            [0.70, 0.02, 0.12],
            [0.75, 0.01, 0.13],
            [0.80, 0.00, 0.14],
            [0.85, -0.01, 0.15],
            [0.90, -0.02, 0.16],
            [0.95, -0.03, 0.17],
            [1.00, -0.04, 0.18],
            [1.05, -0.05, 0.19],
            [1.10, -0.06, 0.20],
        ], dtype=float)

        features = preprocess_landmarks(sample)

        self.assertEqual(features.shape, (63,))
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(np.allclose(features, 0))

    def test_preprocess_landmarks_uses_relative_coordinates(self):
        sample = np.array([
            [0.10, 0.20, 0.00],
            [0.20, 0.20, 0.01],
            [0.30, 0.20, 0.02],
        ], dtype=float)
        with self.assertRaises(ValueError):
            preprocess_landmarks(sample.reshape(-1))


if __name__ == "__main__":
    unittest.main()
