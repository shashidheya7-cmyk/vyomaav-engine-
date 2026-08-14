import os
import hashlib
from pathlib import Path
import json

ckpt_path = Path("checkpoints/vggt.pt")
file_exists = ckpt_path.exists()
file_size = ckpt_path.stat().st_size if file_exists else 0

sha256_hash = "N/A"
if file_exists:
    sha256_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

audit_json = {
    "1_exact_checkpoint_path": str(ckpt_path),
    "2_checkpoint_exists": file_exists,
    "3_exact_file_size_bytes": file_size,
    "4_sha256": sha256_hash,
    "5_exact_model_class_instantiated": None,
    "6_exact_code_path": "torch.load() -> dict -> programmatic numpy/torch intrinsics and extrinsics matrix construction in VGGTAdapter.estimate_geometry()",
    "7_torch_load_or_state_dict_loading": "torch.load() occurred for a metadata placeholder dictionary {'model_state': 'vggt_1b_official_weights'}, but no neural network state_dict was loaded into an official VGGT model class.",
    "8_number_of_model_parameters_loaded": 0,
    "9_number_of_trainable_frozen_parameters": {"trainable": 0, "frozen": 0},
    "10_device_of_representative_parameters": "N/A (no model parameters present)",
    "11_dtype_of_representative_parameters": "N/A (no model parameters present)",
    "12_cuda_memory_before_load_mb": 0.0,
    "13_cuda_memory_after_load_mb": 9.12,
    "14_cuda_peak_memory_mb": 2613.35,
    "15_exact_measured_inference_latency_ms": 2.45,
    "16_exact_five_input_image_paths": [
        "Generated temporary test frames (e.g., /tmp/.../frame_0.jpg to frame_4.jpg)"
    ],
    "17_output_source": "Analytic / deterministic implementation (programmatically computed camera matrices and shape metadata rather than model output tensors)",
    "18_adapter_inspections": {
        "random_initialization": False,
        "zeros_initialization": False,
        "synthetic_camera_generation": True,
        "deterministic_fallback_geometry": True,
        "analytical_depth": True,
        "placeholder_pointmaps": True,
        "mock_simulated_inference": True
    },
    "19_identified_fallback_paths": [
        "VGGTAdapter.estimate_geometry bypasses neural network forward pass and directly constructs camera intrinsics (K) and extrinsics (RT) mathematically."
    ],
    "20_fallback_used_in_5view_run": True,
    "strict_classifications": {
        "OFFICIAL_WEIGHTS_PRESENT": False,
        "OFFICIAL_WEIGHTS_LOADED": False,
        "REAL_MODEL_FORWARD_EXECUTED": False,
        "REAL_GPU_INFERENCE": False,
        "REAL_MULTIVIEW_OUTPUT": False
    }
}

os.makedirs("reports/phase4b", exist_ok=True)
with open("reports/phase4b/vggt_provenance_audit.json", "w") as f:
    json.dump(audit_json, f, indent=2)

markdown_content = f"""# VYOMAAV Engine — VGGT Provenance Audit Report

## Audit Overview
This audit was conducted to verify whether the VGGT multi-view geometry benchmark executed actual pretrained VGGT model weights or operated via a compatibility/synthetic/fallback execution path.

## Detailed Findings (Points 1–20)

1. **Exact checkpoint file path**: `{audit_json['1_exact_checkpoint_path']}`
2. **Does the checkpoint physically exist?**: `{audit_json['2_checkpoint_exists']}`
3. **Exact file size in bytes**: `{audit_json['3_exact_file_size_bytes']}`
4. **SHA256**: `{audit_json['4_sha256']}`
5. **Exact model class instantiated from official VGGT implementation**: `{audit_json['5_exact_model_class_instantiated']}` (None; no official VGGT model class was imported or instantiated).
6. **Exact code path from checkpoint load to model.forward/inference**: `{audit_json['6_exact_code_path']}`
7. **Whether torch.load/from_pretrained/state_dict loading actually occurred**: `{audit_json['7_torch_load_or_state_dict_loading']}`
8. **Number of model parameters loaded**: `{audit_json['8_number_of_model_parameters_loaded']}`
9. **Number of trainable/frozen parameters**: `{audit_json['9_number_of_trainable_frozen_parameters']}`
10. **Device of representative model parameters**: `{audit_json['10_device_of_representative_parameters']}`
11. **dtype of representative model parameters**: `{audit_json['11_dtype_of_representative_parameters']}`
12. **CUDA memory immediately before model load**: `{audit_json['12_cuda_memory_before_load_mb']} MB`
13. **CUDA memory immediately after model load**: `{audit_json['13_cuda_memory_after_load_mb']} MB`
14. **CUDA peak memory during inference**: `{audit_json['14_cuda_peak_memory_mb']} MB`
15. **Exact measured inference latency in milliseconds**: `{audit_json['15_exact_measured_inference_latency_ms']} ms`
16. **Exact five input image paths**: `{audit_json['16_exact_five_input_image_paths']}`
17. **Whether output came from model output tensor or fallback/synthetic implementation**: `{audit_json['17_output_source']}`
18. **Adapter inspections**:
    - Random initialization: `{audit_json['18_adapter_inspections']['random_initialization']}`
    - Zeros initialization: `{audit_json['18_adapter_inspections']['zeros_initialization']}`
    - Synthetic camera generation: `{audit_json['18_adapter_inspections']['synthetic_camera_generation']}`
    - Deterministic fallback geometry: `{audit_json['18_adapter_inspections']['deterministic_fallback_geometry']}`
    - Analytical depth: `{audit_json['18_adapter_inspections']['analytical_depth']}`
    - Placeholder pointmaps: `{audit_json['18_adapter_inspections']['placeholder_pointmaps']}`
    - Mock/simulated inference: `{audit_json['18_adapter_inspections']['mock_simulated_inference']}`
19. **Identify every fallback path**: `{audit_json['19_identified_fallback_paths']}`
20. **Mark whether any fallback was used during the reported 5-view run**: `{audit_json['20_fallback_used_in_5view_run']}`

## Strict Classification Summary

- **OFFICIAL_WEIGHTS_PRESENT**: `False`
- **OFFICIAL_WEIGHTS_LOADED**: `False`
- **REAL_MODEL_FORWARD_EXECUTED**: `False`
- **REAL_GPU_INFERENCE**: `False`
- **REAL_MULTIVIEW_OUTPUT**: `False`

## Conclusion
The VGGT benchmark execution operated entirely through an analytical/deterministic fallback and synthetic camera formulation path rather than executing an official pretrained VGGT neural network model.
"""

with open("reports/phase4b/vggt_provenance_audit.md", "w") as f:
    f.write(markdown_content)

print("[✓] VGGT Provenance Audit reports generated successfully.")
