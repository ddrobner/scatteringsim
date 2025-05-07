from dataclasses import dataclass
from numpy import float32

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


@dataclass
class ScatteredDeposit:
    alpha_indices: list[int]
    alpha_energies: list[float32]
    proton_energies: list[float32]
    particle_id: int

    def __post_init__(self):
        if self.proton_energies is None:
            self.proton_energies = []
    
    @property
    def num_scatters(self):
        return len(self.proton_energies)