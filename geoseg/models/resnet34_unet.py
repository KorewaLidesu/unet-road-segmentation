"""GohVh's ResNet34-UNet, imported from its pinned upstream checkout.

https://github.com/GohVh/resnet34-unet ships no licence, so its code is not
copied into this repo. It is a git submodule at ``third_party/resnet34-unet``,
pinned to the commit this pipeline was built against, and ``UnetResnet34`` is
loaded from there unmodified: an ImageNet-pretrained torchvision ResNet34
encoder, one extra 1024-channel bridge, five transposed-conv decoder blocks.

Inputs must be a multiple of 64 on each side -- the bridge pools once more than
the encoder, and the decoder's skip concatenations need the sizes to line up.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

UPSTREAM_URL = 'https://github.com/GohVh/resnet34-unet'
UPSTREAM_DIR = Path(os.environ.get(
    'RESNET34_UNET_DIR',
    Path(__file__).resolve().parents[2] / 'third_party' / 'resnet34-unet'))


def _load_upstream():
    model_py = UPSTREAM_DIR / 'model.py'
    if not model_py.exists():
        raise FileNotFoundError(
            f'{model_py} not found. The model comes from a git submodule; fetch it with\n'
            f'    git submodule update --init\n'
            f'or, in a copy of this repo without git metadata,\n'
            f'    git clone {UPSTREAM_URL} {UPSTREAM_DIR}')
    # A unique module name: upstream's file is called plain "model".
    spec = importlib.util.spec_from_file_location('gohvh_resnet34_unet', model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ResNet34UNet(num_classes=2):
    """Upstream's ``UnetResnet34``. Downloads the ImageNet ResNet34 weights on first use."""
    return _load_upstream().UnetResnet34(num_classes=num_classes)


def upstream_sha() -> str | None:
    """Commit of the upstream checkout, or None if it is not a git checkout."""
    # Without its own .git, `git -C` would walk up and report this repo's HEAD.
    if not (UPSTREAM_DIR / '.git').exists():
        return None
    try:
        out = subprocess.run(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None
