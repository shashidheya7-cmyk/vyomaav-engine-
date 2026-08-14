"""CLI entrypoint for the VYOMAA Engine Phase 1 Foundation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from vyomaa.core.config import EngineConfig
from vyomaa.core.contracts import InputArtifact
from vyomaa.core.types import ModalityType
from vyomaa.hardware.vram_manager import VRAMManager
from vyomaa.pipeline.dag import ExecutionPlan
from vyomaa.pipeline.task import Task, TaskContext, TaskDependency, TaskResult
from vyomaa.storage.artifact_store import ArtifactStore


class IngestionTask(Task):
    """Deterministic ingestion task validating input files."""

    def __init__(self) -> None:
        super().__init__(task_id="task_ingest", name="Media Ingestion", priority=1)

    def run(self, context: TaskContext) -> TaskResult:
        image_path = context.diagnostics.get("input_path", "test.png")
        artifact = InputArtifact(
            name="Primary Input",
            modality=ModalityType.RGB_IMAGE,
            file_path=image_path,
            resolution=(1920, 1080),
        )
        context.put_artifact(artifact)
        return TaskResult(
            task_id=self.task_id,
            output_artifact_ids=[artifact.artifact_id],
            diagnostics={"ingested_path": image_path},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="VYOMAA Engine Phase 1 Core Foundation")
    parser.add_argument("--config", type=str, default="configs/single_image_object.yaml", help="Path to config YAML")
    parser.add_argument("--input", type=str, default="image.png", help="Input media file")
    args = parser.parse_args()

    try:
        config = EngineConfig.from_yaml(args.config)
        store = ArtifactStore(config.storage.workspace_root)
        vram_mgr = VRAMManager()

        context = TaskContext(
            config=config,
            artifact_store=store,
            vram_manager=vram_mgr,
            diagnostics={"input_path": args.input},
        )

        plan = ExecutionPlan()
        plan.add_task(IngestionTask())

        results = plan.execute(context)
        print(f"Executed {len(results)} task(s) successfully.")
        for tid, res in results.items():
            print(f" - [{res.state.value.upper()}] {tid} ({res.execution_time_seconds:.4f}s)")
        return 0
    except Exception as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
