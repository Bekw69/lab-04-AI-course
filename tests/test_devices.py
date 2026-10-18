"""Device selection must handle Intel and CPU-only PyTorch installations."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lab04_core import PROMPT, pick_device, vlm_extract


class DeviceTests(unittest.TestCase):
    def device(self, *, cuda=False, xpu=False, mps=False, has_xpu=True):
        available = lambda value: SimpleNamespace(is_available=lambda: value)
        torch = SimpleNamespace(cuda=available(cuda), backends=SimpleNamespace(mps=available(mps)))
        if has_xpu:
            torch.xpu = available(xpu)
        with patch.dict(sys.modules, {'torch': torch}):
            return pick_device()

    def test_prefers_cuda_when_both_accelerators_exist(self):
        self.assertEqual(self.device(cuda=True, xpu=True), 'cuda')

    def test_selects_intel_xpu_without_cuda(self):
        self.assertEqual(self.device(xpu=True), 'xpu')

    def test_selects_mps_without_intel_support(self):
        self.assertEqual(self.device(mps=True, has_xpu=False), 'mps')

    def test_cpu_is_a_valid_fallback(self):
        self.assertEqual(self.device(), 'cpu')
        self.assertEqual(self.device(has_xpu=False), 'cpu')

    def test_openvino_receives_the_same_prompt_and_hint_without_torch(self):
        calls = []
        model = SimpleNamespace(backend='openvino', extract=lambda path, **kw: calls.append((path, kw)) or '{}')
        with patch.dict(sys.modules, {'torch': None}):
            raw = vlm_extract(model, None, 'invoice.png', hint='OCR observation')
        self.assertEqual(raw, '{}')
        self.assertEqual(calls, [('invoice.png', {
            'prompt': PROMPT + '\n\nOCR text of the same image (may contain errors):\nOCR observation',
            'max_new_tokens': 512,
        })])


if __name__ == '__main__':
    unittest.main()
