from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from vyomaa.multiview.contracts import ViewSet, GeometryEvidence

class BaseGeometryBackend(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = getattr(torch, config.get("dtype", "float32"))
        self.batch_limit = config.get("batch_limit", 4)
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def estimate_geometry(self, view_set: ViewSet) -> GeometryEvidence:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        pass

# Alias for backward compatibility
BaseGeometryAdapter = BaseGeometryBackend
