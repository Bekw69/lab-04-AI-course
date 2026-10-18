"""Optional INT4 OpenVINO runtime for Intel GPUs and CPUs.

The quantized export is a separate experiment, not the float16 reference model.
Only model data is downloaded; no code from its Hugging Face repository is run.
"""

from importlib.metadata import version
from pathlib import Path

from lab04_core import MODEL_ID

INTEL_MODEL_ID = 'helenai/Qwen2-VL-2B-Instruct-ov-int4'
INTEL_REVISION = '3264f1a456ab58768dbeb514dcb348e4af16c392'


def pick_intel_device(requested, devices, names):
    """Use the validated CPU path for Iris Xe; do not silently use its bad GPU path."""
    selected = requested.upper()
    if selected == 'AUTO':
        candidate = next((d for d in devices if d.startswith('GPU')), 'CPU')
        selected = 'CPU' if 'iris' in names.get(candidate, '').lower() else candidate
    if selected not in devices:
        raise RuntimeError(f'OpenVINO device {selected} is unavailable; found: {devices}')
    if selected.startswith('GPU') and 'iris' in names[selected].lower():
        raise RuntimeError(
            'Iris Xe was detected, but this model failed GPU accuracy/kernel checks on this computer. '
            'Use --device AUTO or --device CPU for the validated OpenVINO CPU path.'
        )
    return selected


class IntelVLM:
    backend = 'openvino'
    dtype = 'INT4 language / INT8 vision and embeddings'

    def __init__(self, device='AUTO', model_dir='.tools/models/qwen2-vl-2b-int4'):
        import openvino as ov
        import openvino_genai
        from huggingface_hub import snapshot_download

        core = ov.Core()
        devices = core.available_devices
        names = {d: core.get_property(d, 'FULL_DEVICE_NAME') for d in devices}
        requested = pick_intel_device(device, devices, names)
        self.device = requested
        self.device_name = core.get_property(requested, 'FULL_DEVICE_NAME')
        self.model_dir = Path(model_dir)
        snapshot_download(
            INTEL_MODEL_ID, revision=INTEL_REVISION, local_dir=self.model_dir,
            allow_patterns=['*.xml', '*.bin', '*.json', '*.txt', '*.jinja', 'README.md'],
            max_workers=2, token=False,
        )
        properties = {'CACHE_DIR': '.tools/openvino-cache'} if requested.startswith('GPU') else {
            'INFERENCE_PRECISION_HINT': 'f32',  # Same as the verified i5 CPU default.
        }
        self.pipeline = openvino_genai.VLMPipeline(str(self.model_dir), requested, **properties)

    def extract(self, image_path, *, prompt, max_new_tokens=512):
        import numpy as np
        import openvino as ov
        from PIL import Image

        with Image.open(image_path) as image:
            # GenAI's image input uses NHWC, including a batch dimension.
            rgb = np.array(image.convert('RGB'))[None]
        result = self.pipeline.generate(
            prompt, images=[ov.Tensor(rgb)],
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        metrics = getattr(result, 'perf_metrics', None)
        self.last_metrics = {}
        for label, method in (('input_tokens', 'get_num_input_tokens'),
                              ('generated_tokens', 'get_num_generated_tokens')):
            getter = getattr(metrics, method, None)
            if getter is not None:
                self.last_metrics[label] = int(getter())
        return result.texts[0]

    def metadata(self):
        return {
            'backend': self.backend, 'base_model_id': MODEL_ID,
            'model_id': INTEL_MODEL_ID, 'model_revision': INTEL_REVISION,
            'device': self.device, 'device_name': self.device_name,
            'dtype': self.dtype, 'quantized': True,
            'activation_precision_hint': 'FP32' if self.device == 'CPU' else 'device default',
            'device_policy': 'AUTO uses the validated CPU path on Iris Xe',
            'versions': {name: version(name) for name in
                         ('openvino', 'openvino-genai', 'openvino-tokenizers', 'huggingface-hub', 'Pillow', 'numpy')},
        }
