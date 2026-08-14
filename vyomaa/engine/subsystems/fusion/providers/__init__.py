
"""Built-in sparse-view fusion providers; imported for registry side effects."""

from .hunyuan3d import Hunyuan3DProvider
from .instantmesh import InstantMeshProvider
from .openlrm import OpenLRMProvider
from .trellis import TRELLISProvider

__all__ = ["Hunyuan3DProvider", "InstantMeshProvider", "OpenLRMProvider", "TRELLISProvider"]


