"""Generate Task 18 variants on the byte-verified original ring caches."""

from .e01b import file_hash
from .e02_capgeom import trial
from .e02_capfix_ring import METHODS_DIR
from .methods.v3_capenergy_engine import EnergyCapEngine, capture_initial_caps


def _engine(exp, device):
    return EnergyCapEngine(exp.root / "cache/models", exp.geometry, exp.engine_config, device=device, lut_path=exp.lut)


def run_candidate(experiment, method, device, *, seed=42, source=None):
    parameters = experiment.parameters(method)
    files = ["v3_capenergy_engine.py", "v3_ring_midscale_cap.py", "v3_uniform_texture.py",
             "v3_captexture.py", "v3_capgeom_init.py"]
    if parameters["match_energy"]:
        files.append("v3_cap_energy.py")
    if parameters["differential"]:
        files.append("v3_capenergy_differential.py")
    implementation = {file: file_hash(METHODS_DIR / file) for file in files}
    trial(experiment, method, parameters, device, seed=seed, source=source, engine_factory=_engine,
          implementation=implementation, capture=capture_initial_caps)
