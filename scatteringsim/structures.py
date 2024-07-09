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