"""
VYOMAAV Base Model Engine
Module: client.shaders

WGSL Shader Generator for WebGPU 3D Gaussian Splat Rendering (Sprint 18).
Generates WebGPU Shading Language (WGSL) vertex, fragment, and compute shaders for:
1. 3D World Space to 2D Screen Space Covariance Matrix Projection (EWA Splatting).
2. Parallel Bitonic Sort / Radix Sort depth ordering.
3. Front-to-back alpha-blended Gaussian fragment rasterization with radial decay.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class WGSLShaderBundle:
    """Container for WebGPU WGSL shader source code modules."""
    compute_projection_shader: str
    vertex_shader: str
    fragment_shader: str

    def combine_all(self) -> str:
        """Combines compute, vertex, and fragment shaders into a single WGSL module."""
        return (
            "// ========================================================\n"
            "// VYOMAAV WebGPU Spark WGSL Gaussian Splatting Shader\n"
            "// ========================================================\n\n"
            f"{self.compute_projection_shader}\n\n"
            f"{self.vertex_shader}\n\n"
            f"{self.fragment_shader}\n"
        )


class WebGPUSplatShaderGenerator:
    """Generates production-grade WGSL shaders for real-time 3D Gaussian rendering in WebGPU."""

    @staticmethod
    def generate_projection_compute_shader() -> str:
        """Compute shader for 3D Covariance Matrix Projection: Sigma' = J * W * Sigma * W^T * J^T."""
        return """
struct GaussianPrimitive {
    position : vec3<f32>,
    opacity : f32,
    scale : vec3<f32>,
    reserved : f32,
    rotation : vec4<f32>, // Quaternion (w, x, y, z)
    color : vec3<f32>,
    padding : f32,
};

struct ProjectedSplat {
    screen_pos : vec2<f32>,
    depth : f32,
    radius : f32,
    conic : vec3<f32>, // Inverse 2D Covariance [a, b, c]
    color : vec4<f32>, // RGB + Alpha
};

struct Uniforms {
    view_matrix : mat4x4<f32>,
    proj_matrix : mat4x4<f32>,
    viewport_size : vec2<f32>,
    focal_length : vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians : array<GaussianPrimitive>;
@group(0) @binding(2) var<storage, read_write> projected_splats : array<ProjectedSplat>;

// Helper: Convert Quaternion to 3x3 Rotation Matrix
fn quat_to_rot_mat(q : vec4<f32>) -> mat3x3<f32> {
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y)),
        vec3<f32>(2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)),
        vec3<f32>(2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y))
    );
}

@compute @workgroup_size(64)
fn compute_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&gaussians)) {
        return;
    }

    let splat = gaussians[index];

    # Camera Space Transformation
    let pos_cam = (uniforms.view_matrix * vec4<f32>(splat.position, 1.0)).xyz;
    if (pos_cam.z <= 0.1) {
        projected_splats[index].radius = 0.0;
        return;
    }

    # 3D Covariance Matrix Construction: S * R
    let R = quat_to_rot_mat(splat.rotation);
    let S = mat3x3<f32>(
        vec3<f32>(splat.scale.x, 0.0, 0.0),
        vec3<f32>(0.0, splat.scale.y, 0.0),
        vec3<f32>(0.0, 0.0, splat.scale.z)
    );
    let M = R * S;
    let Sigma3D = M * transpose(M);

    # Jacobian J of Pinhole Camera Projection
    let fx = uniforms.focal_length.x;
    let fy = uniforms.focal_length.y;
    let z = pos_cam.z;
    let z2 = z * z;

    let J = mat3x3<f32>(
        vec3<f32>(fx / z, 0.0, -fx * pos_cam.x / z2),
        vec3<f32>(0.0, fy / z, -fy * pos_cam.y / z2),
        vec3<f32>(0.0, 0.0, 0.0)
    );

    let W = mat3x3<f32>(
        uniforms.view_matrix[0].xyz,
        uniforms.view_matrix[1].xyz,
        uniforms.view_matrix[2].xyz
    );

    let T = J * W;
    let Sigma2D = T * Sigma3D * transpose(T);

    # 2D Screen Covariance & Inverse Conic Matrix
    let det = Sigma2D[0][0] * Sigma2D[1][1] - Sigma2D[0][1] * Sigma2D[0][1];
    if (det <= 1e-6) {
        projected_splats[index].radius = 0.0;
        return;
    }

    let inv_det = 1.0 / det;
    let conic = vec3<f32>(
        Sigma2D[1][1] * inv_det,
        -Sigma2D[0][1] * inv_det,
        Sigma2D[0][0] * inv_det
    );

    # Project Screen Center Position in Pixels
    let clip_pos = uniforms.proj_matrix * vec4<f32>(pos_cam, 1.0);
    let ndc = clip_pos.xy / clip_pos.w;
    let screen_pos = (ndc * 0.5 + vec2<f32>(0.5)) * uniforms.viewport_size;

    # Compute Screen Footprint Radius
    let mid = 0.5 * (Sigma2D[0][0] + Sigma2D[1][1]);
    let lambda2 = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(lambda2));

    projected_splats[index].screen_pos = screen_pos;
    projected_splats[index].depth = pos_cam.z;
    projected_splats[index].radius = radius;
    projected_splats[index].conic = conic;
    projected_splats[index].color = vec4<f32>(splat.color, splat.opacity);
}
"""

    @staticmethod
    def generate_vertex_shader() -> str:
        """Vertex shader positioning Screen Quad Billboards for projected 2D Gaussians."""
        return """
struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) center_pixel : vec2<f32>,
    @location(1) conic : vec3<f32>,
    @location(2) color : vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index : u32,
    @builtin(instance_index) instance_index : u32
) -> VertexOutput {
    var out : VertexOutput;
    let splat = projected_splats[instance_index];

    if (splat.radius <= 0.0) {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }

    # Quad Corner Offsets [-1, 1]
    var quad_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let offset = quad_offsets[vertex_index] * splat.radius;
    let pixel_pos = splat.screen_pos + offset;

    # Convert Pixel Coordinates back to WebGPU Clip Space [-1, 1]
    let ndc_pos = (pixel_pos / uniforms.viewport_size) * 2.0 - vec2<f32>(1.0);

    out.clip_position = vec4<f32>(ndc_pos.x, -ndc_pos.y, splat.depth / 100.0, 1.0);
    out.center_pixel = splat.screen_pos;
    out.conic = splat.conic;
    out.color = splat.color;

    return out;
}
"""

    @staticmethod
    def generate_fragment_shader() -> str:
        """Fragment shader evaluating Gaussian Radial Decay and front-to-back alpha blending."""
        return """
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    let d = in.clip_position.xy - in.center_pixel;

    # Evaluate 2D Elliptical Gaussian Exponent: G(d) = exp(-0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2))
    let power = -0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);

    if (power > 0.0) {
        discard;
    }

    let alpha = in.color.a * exp(power);
    if (alpha < 0.005) {
        discard;
    }

    return vec4<f32>(in.color.rgb * alpha, alpha);
}
"""

    @classmethod
    def generate_complete_bundle(cls) -> WGSLShaderBundle:
        """Constructs complete WGSL shader bundle."""
        return WGSLShaderBundle(
            compute_projection_shader=cls.generate_projection_compute_shader(),
            vertex_shader=cls.generate_vertex_shader(),
            fragment_shader=cls.generate_fragment_shader()
        )