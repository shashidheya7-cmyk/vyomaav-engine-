"""3D Generation Worker Subsystem."""

from .base_worker import Base3DWorker
from .triposr_worker import TripoSRWorker, TRIPOSR_SPEC
from .trellis_worker import TRELLISWorker, TRELLIS_SPEC
from .hunyuan3d_worker import Hunyuan3DWorker, HUNYUAN3D_SPEC

__all__ = [
    "Base3DWorker",
    "TripoSRWorker",
    "TRIPOSR_SPEC",
    "TRELLISWorker",
    "TRELLIS_SPEC",
    "Hunyuan3DWorker",
    "HUNYUAN3D_SPEC",
]
