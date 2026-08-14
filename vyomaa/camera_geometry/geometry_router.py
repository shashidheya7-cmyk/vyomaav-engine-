import logging
from typing import Dict, Any, Optional
from vyomaa.multiview.contracts import ViewSet, GeometryEvidence
from vyomaa.camera_geometry.base import BaseGeometryBackend

logger = logging.getLogger("vyomaa.camera_geometry.geometry_router")

class GeometryRouter:
    def __init__(self, backends: Dict[str, BaseGeometryBackend], policy_config: Optional[Dict[str, Any]] = None):
        self.backends = backends
        self.policy_config = policy_config or {
            "primary": "VGGT",
            "analytic_fallback": "analytic_fallback",
            "confidence_threshold": 0.75
        }

    def route(self, view_set: ViewSet) -> GeometryEvidence:
        primary_name = self.policy_config.get("primary", "VGGT")
        vggt_backend = self.backends.get(primary_name) or self.backends.get("vggt")
        
        if vggt_backend and vggt_backend.is_available():
            logger.info(f"Routing to primary geometry backend: {primary_name}")
            evidence = vggt_backend.estimate_geometry(view_set)
            evidence.provenance["routing_decision"] = "primary_VGGT"
            return evidence

        fallback_name = self.policy_config.get("analytic_fallback", "analytic_fallback")
        fallback_backend = self.backends.get(fallback_name)
        if fallback_backend and fallback_backend.is_available():
            logger.warning("Primary VGGT backend unavailable. Executing fallback geometry.")
            evidence = fallback_backend.estimate_geometry(view_set)
            evidence.provenance["routing_decision"] = "fallback_analytic_geometry"
            return evidence

        raise RuntimeError("No available geometry backends could process the ViewSet.")
