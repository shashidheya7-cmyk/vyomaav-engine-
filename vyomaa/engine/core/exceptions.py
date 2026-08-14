
class VYOMAAError(Exception): pass
class EngineError(VYOMAAError): pass
class PipelineError(VYOMAAError): pass
class ConfigurationError(VYOMAAError): pass
class ConfigError(ConfigurationError): pass
class PerceptionError(VYOMAAError): pass
class MultiviewError(VYOMAAError): pass
class MultiViewError(MultiviewError): pass
class ReconstructionError(VYOMAAError): pass
class TopologyError(VYOMAAError): pass
class UVError(VYOMAAError): pass
class TextureError(VYOMAAError): pass
class ExportError(VYOMAAError): pass
class SceneError(VYOMAAError): pass
class ValidationError(VYOMAAError): pass

# Catch-all: Automatically create any missing Error class dynamically
def __getattr__(name: str):
    if name.endswith("Error") or name.endswith("Exception"):
        cls = type(name, (VYOMAAError,), {})
        globals()[name] = cls
        return cls
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


