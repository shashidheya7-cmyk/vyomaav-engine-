"""
VYOMAAV Base Model Engine
Test Suite: tests/test_cli.py

Pytest suite validating Sprint 16: CLI argument parsing, full pipeline orchestration,
subcommand dispatching, and manifest file generation.
"""

import os
import json
import pytest
from cli.vyomaav import VYOMAAVPipelineRunner, main, build_parser
from engine.compiler import CompilationTarget


def test_cli_argument_parser():
    parser = build_parser()
    parsed = parser.parse_args(["pipeline", "--scene-id", "TestScene", "--target", "webgpu_spark"])

    assert parsed.command == "pipeline"
    assert parsed.scene_id == "TestScene"
    assert parsed.target == "webgpu_spark"


def test_vyomaav_pipeline_runner_end_to_end(tmp_path):
    output_dir = str(tmp_path / "vgc_build")
    runner = VYOMAAVPipelineRunner(scene_id="EndToEndScene")

    # Step 1: Ingestion & SOMG fusion
    scene = runner.run_ingestion_and_fusion(num_frames=2)
    assert len(scene.resolve_active_graph().nodes) > 0

    # Step 2: World Compilation
    manifest_path = runner.compile_world(output_dir=output_dir, target=CompilationTarget.WEBGPU_SPARK)
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert manifest["scene_id"] == "EndToEndScene"
    assert manifest["target"] == "webgpu_spark"

    # Step 3: Simulation & Pathfinding
    sim_res = runner.run_simulation(start_pos=[0.0, 0.2, 0.0], target_pos=[1.5, 0.2, 1.5])
    assert "path_found" in sim_res
    assert sim_res["html_app_bytes"] > 0


def test_main_cli_execution_pipeline_command(tmp_path):
    output_dir = str(tmp_path / "cli_out")
    cli_args = [
        "pipeline",
        "--scene-id", "CLIScene",
        "--output-dir", output_dir,
        "--target", "webgpu_spark"
    ]

    exit_code = main(cli_args)
    assert exit_code == 0
    assert os.path.exists(os.path.join(output_dir, "vgc_manifest.json"))