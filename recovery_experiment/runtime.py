from __future__ import annotations
import os
import sys
from pathlib import Path


def bootstrap(reference=None):
    """Use the in-repository task; reference is retained for old CLI compatibility."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ.pop('DISPLAY', None)
    os.environ['MUJOCO_GL'] = 'egl'
    os.chdir(root)
    return root
