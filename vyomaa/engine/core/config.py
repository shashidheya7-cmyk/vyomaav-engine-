
"""Configuration model and YAML loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class EngineConfig:
    """Validated runtime configuration for a VYOMAA pipeline."""

    device: str = "cuda"
    precision: str = "fp16"
    auto_vram_cleanup: bool = True
    pipeline_stages: list[str] = field(default_factory=lambda: ["perception", "multiview", "fusion", "geometry", "topology", "uv", "texture", "material", "export"])
    perception: dict[str, Any] = field(default_factory=dict)
    multiview: dict[str, Any] = field(default_factory=dict)
    fusion: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    uv: dict[str, Any] = field(default_factory=dict)
    texture: dict[str, Any] = field(default_factory=dict)
    material: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.precision not in {"fp16", "fp32", "bf16"}:
            raise ConfigurationError("precision must be one of fp16, fp32, bf16")
        if not isinstance(self.fusion, dict) or not self.fusion.get("provider"):
            raise ConfigurationError("fusion.provider must be configured")
        if not self.pipeline_stages:
            raise ConfigurationError("pipeline_stages must not be empty")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EngineConfig":
        """Load and validate an engine configuration from a YAML document."""
        path = Path(yaml_path)
        if not path.is_file():
            raise ConfigurationError(f"configuration file does not exist: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream) if yaml is not None else _load_basic_yaml(stream.read())
                data = data or {}
        except (OSError, ValueError, getattr(yaml, "YAMLError", ValueError)) as exc:
            raise ConfigurationError(f"unable to load configuration: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigurationError("configuration root must be a mapping")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError(f"unsupported configuration field: {exc}") from exc


def _load_basic_yaml(text: str) -> dict[str, Any]:
    """Parse the mapping/list/scalar YAML subset used by the shipped config.

    PyYAML remains the production parser; this dependency-free parser preserves a
    runnable verification entrypoint in constrained CPU-only environments.
    """
    lines = [(len(line) - len(line.lstrip()), line.strip()) for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]

    def scalar(value: str) -> Any:
        if value in {"true", "True"}: return True
        if value in {"false", "False"}: return False
        if value in {"null", "Null", "~"}: return None
        if re.fullmatch(r"[-+]?\d+", value): return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value): return float(value)
        return value.strip("\"'")

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines) or lines[index][0] < indent: return {}, index
        is_list = lines[index][1].startswith("- ")
        result: Any = [] if is_list else {}
        while index < len(lines) and lines[index][0] == indent:
            _, content = lines[index]
            if is_list:
                if not content.startswith("- "): break
                result.append(scalar(content[2:].strip()))
                index += 1
                continue
            if ":" not in content: raise ValueError(f"invalid YAML mapping line: {content}")
            key, value = (part.strip() for part in content.split(":", 1))
            index += 1
            if value:
                result[key] = scalar(value)
            elif index < len(lines) and lines[index][0] > indent:
                result[key], index = parse_block(index, lines[index][0])
            else:
                result[key] = {}
        return result, index

    parsed, _ = parse_block(0, 0)
    if not isinstance(parsed, dict): raise ValueError("YAML root must be a mapping")
    return parsed


