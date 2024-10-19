from dataclasses import dataclass

@dataclass
class ScatterFrame:
    e_alpha: float
    e_proton: float
    scattering_angle: float

@dataclass
class AlphaEvent:
    alpha_path: list[float]
    proton_scatters: list[float]
    scatter_energy: list[ScatterFrame]

class TrackedScatter(float):
    def __init__(self, val, scatter):
        self._scatter = scatter
        super(float, val)

    def __new__(cls, val):
        return float.__new__(cls, float(val))