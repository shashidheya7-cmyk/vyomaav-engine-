
"""CLI entrypoint for a complete VYOMAA image-to-GLB generation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vyomaa"))

from engine.core.config import EngineConfig
from engine.core.exceptions import VyomaaEngineError
from engine.pipeline.orchestrator import Orchestrator
from engine.scene.scene import Scene


def parse_args() -> argparse.Namespace:
    """Parse the required input image, output GLB, and optional YAML config."""
    parser = argparse.ArgumentParser(description="VYOMAA: production image-to-3D GLB generation")
    parser.add_argument("--image", required=True, type=Path, help="Input PNG/JPEG image")
    parser.add_argument("--output", required=True, type=Path, help="Destination .glb asset")
    parser.add_argument("--config", type=Path, default=ROOT / "vyomaa" / "configs" / "engine_config.yaml", help="Engine YAML configuration")
    return parser.parse_args()


def main() -> int:
    """Run the complete nine-stage pipeline and print machine-readable diagnostics."""
    arguments = parse_args()
    if not arguments.image.is_file():
        print(f"error: input image does not exist: {arguments.image}", file=sys.stderr)
        return 2
    try:
        config = EngineConfig.from_yaml(str(arguments.config))
        scene = Scene(source_image_path=str(arguments.image.resolve()), output_path=str(arguments.output.resolve()))
        output = Orchestrator(config).run(scene)
        print(json.dumps({"output": output.output_path, "diagnostics": output.diagnostics}, indent=2, default=str))
        return 0
    except VyomaaEngineError as exc:
        print(f"VYOMAA pipeline error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


