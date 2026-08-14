from abc import ABC, abstractmethod
import torch

class IVisionEncoder(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ICameraEstimator(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass

class IWorldMemory(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, query: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        pass

class IRelationshipHead(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, entity_features: torch.Tensor) -> torch.Tensor:
        pass

class IBaseModelCore(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        pass
