"""CLI entrypoint for the VYOMAA Phase 2 Geometric Evidence Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from vyomaa.core.config import EngineConfig
from vyomaa.core.contracts import Observation, ValidationReport
from vyomaa.representations.point_cloud import PointCloud
from vyomaa.core.types import ModalityType
from vyomaa.hardware.gpu_info import GPUInfo
from vyomaa.hardware.vram_manager import VRAMManager
from vyomaa.pipeline.dag import ExecutionPlan
from vyomaa.pipeline.task import TaskContext
from vyomaa.pipeline.tasks import (
    IngestionTask,
    VisionPerceptionTask,
    MultiViewEvidenceTask,
    PointMapBackprojectionTask,
    GeometricValidationTask,
)
from vyomaa.storage.artifact_store import ArtifactStore


def run_pipeline(modality: ModalityType, input_path: str, config_path: str) -> int:
    """Build and execute the full Phase 2 Geometric Evidence DAG."""
    print(f"\n=======================================================")
    print(f" VYOMAA PHASE 2 — GEOMETRIC EVIDENCE PIPELINE")
    print(f" Modality: {modality.value.upper()}")
    print(f" Input:    {input_path}")
    print(f" Config:   {config_path}")
    print(f"=======================================================\n")

    p_in = Path(input_path)
    if not p_in.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    # 1. Load Config & Hardware Manager
    config = EngineConfig.from_yaml(config_path) if Path(config_path).is_file() else EngineConfig()
    store = ArtifactStore(config.storage.workspace_root)
    gpu_info = GPUInfo.discover()
    vram_mgr = VRAMManager()

    print(f"[Hardware Discovery] Device: {gpu_info.device_name} (CUDA={gpu_info.is_cuda_available}, Total VRAM={gpu_info.total_vram_gb} GB)")

    # 2. Build Context
    context = TaskContext(
        config=config,
        artifact_store=store,
        vram_manager=vram_mgr,
        diagnostics={
            "input_path": str(p_in.resolve()),
            "modality": modality,
        },
    )

    # 3. Assemble Phase 2 DAG
    plan = ExecutionPlan()
    plan.add_task(IngestionTask())
    plan.add_task(VisionPerceptionTask())
    plan.add_task(MultiViewEvidenceTask())
    plan.add_task(PointMapBackprojectionTask())
    plan.add_task(GeometricValidationTask())

    # 4. Execute Tasks
    print(f"[Pipeline Planner] Assembled {len(plan._tasks)} topological tasks. Executing...\n")
    results = plan.execute(context)

    for tid, res in results.items():
        status_sym = "✓" if res.state.value == "completed" else "✗"
        print(f" [{status_sym}] {tid:32s} : {res.state.value.upper():9s} ({res.execution_time_seconds:6.4f}s)")
        if res.error_message:
            print(f"     Error: {res.error_message}")

    # 5. Output Summary Reports
    report_id = context.diagnostics.get("validation_report_id")
    pc_id = context.diagnostics.get("primary_pointcloud_id")
    obs_ids = context.diagnostics.get("observation_ids", [])

    print(f"\n=======================================================")
    print(f" PHASE 2 EVIDENCE SUMMARY REPORT")
    print(f"=======================================================")

    if report_id:
        report = context.get_artifact(report_id)
        if isinstance(report, ValidationReport):
            print(f" Overall Quality Score:    {report.overall_quality_score:.3f}")
            print(f" Camera Pose Stability:     {report.camera_pose_stability:.3f}")
            print(f" Reprojection Error:       {report.reprojection_error_pixels:.3f} px")
            print(f" Is Valid Reconstruction:  {report.is_valid}")
            if report.warnings:
                print(f" Warnings ({len(report.warnings)}):")
                for w in report.warnings:
                    print(f"   - {w}")

    if pc_id:
        pc = context.get_artifact(pc_id)
        if isinstance(pc, PointCloud):
            min_b, max_b = pc.compute_bounds()
            print(f"\n [Point Cloud Artifact]")
            print(f" Artifact ID:   {pc.artifact_id}")
            print(f" Total Points:  {pc.point_count:,}")
            print(f" Has Normals:   {pc.normals is not None}")
            print(f" Has Colors:    {pc.colors is not None}")
            print(f" Bounds Extent: Min {min_b.tolist()} -> Max {max_b.tolist()}")

    print(f"\n [Artifact Store Location]")
    print(f" Stored In: {store.artifacts_dir}")
    print(f" Total Manifests Indexed: {len(store.list_manifests())}")
    print(f"=======================================================\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VYOMAA Phase 2 Geometric Evidence Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Processing modality command")

    # Command: image
    parser_img = subparsers.add_parser("image", help="Process single RGB image")
    parser_img.add_argument("--input", required=True, type=str, help="Path to input image file")
    parser_img.add_argument("--config", default="configs/single_image_object.yaml", type=str, help="Config YAML")

    # Command: multiview
    parser_mv = subparsers.add_parser("multiview", help="Process multi-view image collection")
    parser_mv.add_argument("--input", required=True, type=str, help="Directory containing multi-view images")
    parser_mv.add_argument("--config", default="configs/multiview_object.yaml", type=str, help="Config YAML")

    # Command: video
    parser_vid = subparsers.add_parser("video", help="Process monocular RGB video sequence")
    parser_vid.add_argument("--input", required=True, type=str, help="Path to video file")
    parser_vid.add_argument("--config", default="configs/video_scene.yaml", type=str, help="Config YAML")

    args = parser.parse_args()

    if args.command == "image":
        return run_pipeline(ModalityType.RGB_IMAGE, args.input, args.config)
    elif args.command == "multiview":
        return run_pipeline(ModalityType.MULTIVIEW_IMAGE_SET, args.input, args.config)
    elif args.command == "video":
        return run_pipeline(ModalityType.MONOCULAR_VIDEO, args.input, args.config)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
