"""
VYOMAAV Base Model Engine
Module: cli.vyomaav

Unified Command-Line Interface (CLI) & Pipeline Runner (Sprint 16).
Provides a single CLI entry point (`vyomaav`) with subcommands:
- `ingest`: Parse EXIF & raw sensor/video frames into 3D PerceptionObservations.
- `build-somg`: Build and resolve persistent Spatial Object Memory Graphs.
- `compile`: Run VGC World Compiler Passes 1-4 for UE5 Nanite, WebGPU Spark, or USDZ.
- `simulate`: Execute A* NavMesh pathfinding trajectories & generate WebGPU client HTML.
- `pipeline`: Execute end-to-end video-to-world compilation & WebGPU simulation in one pass.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional, Any
import torch

from dataset.exif import EXIFExtractor, CameraMetadata
from dataset.ingest import MultimodalIngestionPipeline, RawSensorSample, BoundingBox2D
from somg.builder import PerceptionObservation, SOMGEntityBuilder
from somg.fusion import TemporalFusionEngine
from somg.scene import SceneState, DeltaLayer
from base_model.model import VYOMAAVBaseModel
from base_model.contracts import ModelInputBatch
from inference.pipeline import BaseModelToSOMGInferencePipeline
from engine.compiler import VGCWorldCompiler, CompilationFlags, CompilationTarget
from engine.physics import RecastNavMeshEngine
from engine.pathfinding import AStarNavMeshPathfinder
from client.spark import WebGPUSparkPlayerController


class VYOMAAVPipelineRunner:
    """Orchestrates end-to-end execution across ingestion, base model, SOMG, compilation, and simulation."""

    def __init__(self, scene_id: str = "VYOMAAV_World", device: str = "cpu"):
        self.scene_id = scene_id
        self.device = torch.device(device)
        self.scene = SceneState(scene_id=scene_id)

    def run_ingestion_and_fusion(self, num_frames: int = 3) -> SceneState:
        """Simulates multimodal ingestion & temporal fusion over a sequence of video frames."""
        meta = EXIFExtractor.create_simulated_metadata(resolution=(1920, 1080), focal_length_mm=24.0)
        pipeline = MultimodalIngestionPipeline(default_depth_m=3.5)
        fusion = TemporalFusionEngine(self.scene, iou_threshold=0.2)

        for f_idx in range(num_frames):
            sample = RawSensorSample(
                sample_id=f"frame_{f_idx:03d}",
                timestamp_s=f_idx * 0.033,
                camera_metadata=meta,
                detections_2d=[
                    BoundingBox2D("table", 1, 0.92, 400.0, 300.0, 1200.0, 800.0),
                    BoundingBox2D("chair", 2, 0.88, 100.0, 400.0, 500.0, 900.0),
                    BoundingBox2D("lamp", 3, 0.85, 600.0, 100.0, 800.0, 400.0)
                ]
            )
            observations = pipeline.process_sample_to_observations(sample)
            fusion.fuse_frame_observations(observations, delta_id=f"delta_{f_idx}")

        return self.scene

    def compile_world(
        self,
        output_dir: str,
        target: CompilationTarget = CompilationTarget.WEBGPU_SPARK
    ) -> str:
        """Executes VGC World Compiler Pass 1-4 for the target platform."""
        flags = CompilationFlags(
            target=target,
            generate_meshlets=True,
            max_meshlet_vertices=64,
            max_meshlet_triangles=126
        )
        compiler = VGCWorldCompiler(flags)
        bundle = compiler.pass4_compile_package(self.scene, output_directory=output_dir, device=self.device)
        return bundle.manifest_filepath

    def run_simulation(
        self,
        start_pos: List[float],
        target_pos: List[float]
    ) -> Dict[str, Any]:
        """Runs A* NavMesh pathfinding and generates WebGPU Spark HTML client."""
        navmesh = RecastNavMeshEngine().generate_navmesh(self.scene, device=self.device)
        pathfinder = AStarNavMeshPathfinder(navmesh)
        path_res = pathfinder.find_path(start_pos, target_pos, device=self.device)

        html_app = WebGPUSparkPlayerController.generate_webgpu_spark_client_html(self.scene_id)

        return {
            "path_found": path_res.found,
            "path_cost": path_res.total_cost if path_res.found else None,
            "num_waypoints": len(path_res.waypoints),
            "html_app_bytes": len(html_app)
        }


def build_parser() -> argparse.ArgumentParser:
    """Constructs command-line argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="vyomaav",
        description="VYOMAAV Base Model Engine - Unified CLI & World Compiler Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Pipeline command (full end-to-end execution)
    p_pipeline = subparsers.add_parser("pipeline", help="Run full video-to-world compilation pipeline")
    p_pipeline.add_argument("--scene-id", type=str, default="DemoScene", help="Unique scene identifier")
    p_pipeline.add_argument("--output-dir", type=str, default="./vgc_output", help="Compiler target output directory")
    p_pipeline.add_argument("--target", type=str, choices=["ue5_nanite_lumen", "webgpu_spark", "usdz_package"], default="webgpu_spark", help="Compilation target")

    # Compile command
    p_compile = subparsers.add_parser("compile", help="Compile existing SOMG scene graph to target package")
    p_compile.add_argument("--scene-id", type=str, default="DemoScene", help="Scene ID")
    p_compile.add_argument("--output-dir", type=str, default="./vgc_output", help="Compiler output directory")
    p_compile.add_argument("--target", type=str, choices=["ue5_nanite_lumen", "webgpu_spark", "usdz_package"], default="webgpu_spark")

    # Simulate command
    p_sim = subparsers.add_parser("simulate", help="Run NavMesh pathfinding and WebGPU client generation")
    p_sim.add_argument("--scene-id", type=str, default="DemoScene", help="Scene ID")
    p_sim.add_argument("--start", nargs=3, type=float, default=[0.0, 0.2, 0.0], help="Start 3D position x y z")
    p_sim.add_argument("--target", nargs=3, type=float, default=[2.0, 0.2, 2.0], help="Target 3D position x y z")

    return parser


def main(sys_args: Optional[List[str]] = None) -> int:
    """CLI Entry point."""
    parser = build_parser()
    args = parser.parse_args(sys_args)

    if not args.command:
        parser.print_help()
        return 0

    target_map = {
        "ue5_nanite_lumen": CompilationTarget.UNREAL_ENGINE_5,
        "webgpu_spark": CompilationTarget.WEBGPU_SPARK,
        "usdz_package": CompilationTarget.USDZ_PACKAGE
    }

    if args.command == "pipeline":
        print(f"[VYOMAAV CLI] Executing full end-to-end pipeline for scene '{args.scene_id}'...")
        runner = VYOMAAVPipelineRunner(scene_id=args.scene_id)
        
        # 1. Ingest & Fuse
        runner.run_ingestion_and_fusion(num_frames=3)
        print(f"[VYOMAAV CLI] Resolved SOMG graph with {len(runner.scene.resolve_active_graph().nodes)} entities.")

        # 2. Compile Target
        target_enum = target_map[args.target]
        manifest_path = runner.compile_world(args.output_dir, target=target_enum)
        print(f"[VYOMAAV CLI] World compilation successful. Manifest emitted: {manifest_path}")

        # 3. Simulate Pathfinding & WebGPU app
        sim_res = runner.run_simulation(start_pos=[0.0, 0.2, 0.0], target_pos=[2.0, 0.2, 2.0])
        print(f"[VYOMAAV CLI] Pathfinding status: {sim_res['path_found']} | Waypoints: {sim_res['num_waypoints']}")
        print("[VYOMAAV CLI] Pipeline execution complete.")

    elif args.command == "compile":
        runner = VYOMAAVPipelineRunner(scene_id=args.scene_id)
        runner.run_ingestion_and_fusion(num_frames=1)
        manifest_path = runner.compile_world(args.output_dir, target=target_map[args.target])
        print(f"[VYOMAAV CLI] Compiled manifest: {manifest_path}")

    elif args.command == "simulate":
        runner = VYOMAAVPipelineRunner(scene_id=args.scene_id)
        runner.run_ingestion_and_fusion(num_frames=2)
        sim_res = runner.run_simulation(start_pos=args.start, target_pos=args.target)
        print(f"[VYOMAAV CLI] Simulation results: {json.dumps(sim_res, indent=2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())