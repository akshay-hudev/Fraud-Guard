from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path as _Path

_path = _Path(__file__).parent / "training" / "scripts" / "run_pipeline.py"
_spec = spec_from_file_location("training_scripts_run_pipeline", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)
