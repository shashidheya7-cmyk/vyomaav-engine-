import json
from pathlib import Path
from typing import Dict, Any

class ArtifactStore:
    def __init__(self, base_dir: str = "export_package"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact_name: str, data: Dict[str, Any]) -> str:
        path = self.base_dir / f"{artifact_name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)

    def load(self, artifact_name: str) -> Dict[str, Any]:
        path = self.base_dir / f"{artifact_name}.json"
        with open(path, "r") as f:
            return json.load(f)
