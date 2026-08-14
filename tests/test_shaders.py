"""
VYOMAAV Base Model Engine
Test Suite: tests/test_shaders.py

Pytest suite validating Sprint 18: WGSL shader generation, compute shader bindings,
vertex quad positioning, and complete WGSL bundle compilation.
"""

import pytest
from client.shaders import WebGPUSplatShaderGenerator, WGSLShaderBundle


def test_wgsl_projection_compute_shader_generation():
    compute_code = WebGPUSplatShaderGenerator.generate_projection_compute_shader()

    assert "struct GaussianPrimitive" in compute_code
    assert "struct ProjectedSplat" in compute_code
    assert "@compute @workgroup_size(64)" in compute_code
    assert "fn quat_to_rot_mat" in compute_code
    assert "Sigma2D" in compute_code


def test_wgsl_vertex_and_fragment_shader_generation():
    vert_code = WebGPUSplatShaderGenerator.generate_vertex_shader()
    frag_code = WebGPUSplatShaderGenerator.generate_fragment_shader()

    assert "@vertex" in vert_code
    assert "fn vs_main" in vert_code
    assert "quad_offsets" in vert_code

    assert "@fragment" in frag_code
    assert "fn fs_main" in frag_code
    assert "discard;" in frag_code


def test_wgsl_complete_bundle_combination():
    bundle = WebGPUSplatShaderGenerator.generate_complete_bundle()

    assert isinstance(bundle, WGSLShaderBundle)
    full_wgsl = bundle.combine_all()

    assert "// VYOMAAV WebGPU Spark WGSL Gaussian Splatting Shader" in full_wgsl
    assert "@compute" in full_wgsl
    assert "@vertex" in full_wgsl
    assert "@fragment" in full_wgsl