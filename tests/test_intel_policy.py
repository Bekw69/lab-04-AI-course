"""The failed Iris Xe GPU control must never become the automatic lab backend."""

import unittest

from lab04_intel import pick_intel_device


class IntelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.devices = ['CPU', 'GPU']
        self.names = {'CPU': 'Intel Core i5-1235U', 'GPU': 'Intel(R) Iris(R) Xe Graphics (iGPU)'}

    def test_auto_iris_uses_validated_cpu(self):
        self.assertEqual(pick_intel_device('AUTO', self.devices, self.names), 'CPU')

    def test_explicit_iris_gpu_explains_the_failed_control(self):
        with self.assertRaisesRegex(RuntimeError, 'Use --device AUTO or --device CPU'):
            pick_intel_device('GPU', self.devices, self.names)

    def test_explicit_cpu_stays_cpu(self):
        self.assertEqual(pick_intel_device('CPU', self.devices, self.names), 'CPU')

    def test_auto_other_gpu_remains_available(self):
        self.names['GPU'] = 'Intel Arc Graphics'
        self.assertEqual(pick_intel_device('AUTO', self.devices, self.names), 'GPU')

    def test_unavailable_gpu_is_not_reported_as_cpu_execution(self):
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            pick_intel_device('GPU', ['CPU'], self.names)


if __name__ == '__main__':
    unittest.main()
