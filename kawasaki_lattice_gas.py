"""
Kawasaki dynamics simulation of a 2D lattice gas.

In Kawasaki dynamics the total number of particles is conserved. At each
Monte Carlo step a random nearest-neighbour pair is chosen; if the two sites
differ (one occupied, one empty) a swap is attempted and accepted via the
Metropolis criterion.

Hamiltonian:  H = -J * sum_{<i,j>} n_i * n_j
where n_i = 1 (occupied) or 0 (empty) and the sum runs over nearest-neighbour
pairs on a periodic square lattice.

Positive J favours clustering (phase separation at low T).
"""

import numpy as np


class KawasakiLatticeGas:
    """2D lattice gas with Kawasaki (conserved) dynamics."""

    def __init__(self, L: int, density: float, J: float = 1.0, T: float = 2.0,
                 seed: int | None = None):
        """
        Parameters
        ----------
        L       : linear size of the square lattice (L x L sites)
        density : fraction of sites that are occupied  (0 < density < 1)
        J       : coupling constant  (J > 0  → attractive interactions)
        T       : temperature in units of J/k_B
        seed    : random seed for reproducibility
        """
        if not 0 < density < 1:
            raise ValueError("density must be between 0 and 1 (exclusive)")

        self.L = L
        self.J = J
        self.T = T
        self.beta = 1.0 / T

        self.rng = np.random.default_rng(seed)

        # Initialise lattice with the requested density
        n_particles = round(density * L * L)
        flat = np.zeros(L * L, dtype=np.int8)
        flat[:n_particles] = 1
        self.rng.shuffle(flat)
        self.lattice = flat.reshape(L, L)

        self.n_particles = n_particles
        self.sweep = 0          # number of completed MC sweeps

    # ------------------------------------------------------------------
    # Physical observables
    # ------------------------------------------------------------------

    def energy(self) -> float:
        """Total lattice energy (vectorised, no double-counting)."""
        lat = self.lattice
        e = -self.J * (
            np.sum(lat * np.roll(lat, 1, axis=0)) +   # vertical bonds
            np.sum(lat * np.roll(lat, 1, axis=1))      # horizontal bonds
        )
        return float(e)

    def density_field(self) -> np.ndarray:
        """Return a copy of the occupation lattice."""
        return self.lattice.copy()

    def structure_factor(self) -> np.ndarray:
        """
        Static structure factor S(q) = |rho(q)|^2 / N,
        useful for detecting phase separation.
        """
        rho_q = np.fft.fft2(self.lattice.astype(float))
        return (np.abs(rho_q) ** 2) / self.lattice.size

    # ------------------------------------------------------------------
    # Monte Carlo moves
    # ------------------------------------------------------------------

    def _delta_energy(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """
        Energy change for swapping neighbouring sites (i1,j1) and (i2,j2).

        For a nearest-neighbour pair the exact result is:

            dE = J * (n1 - n2) * (S1 - S2)

        where S_k is the sum of occupations of the neighbours of site k,
        *excluding* the other site in the pair.
        """
        L = self.L
        lat = self.lattice
        n1 = int(lat[i1, j1])
        n2 = int(lat[i2, j2])

        if n1 == n2:
            return 0.0

        # Neighbours of site 1, excluding site 2
        s1 = (int(lat[(i1 - 1) % L, j1]) + int(lat[(i1 + 1) % L, j1]) +
              int(lat[i1, (j1 - 1) % L]) + int(lat[i1, (j1 + 1) % L]) -
              n2)

        # Neighbours of site 2, excluding site 1
        s2 = (int(lat[(i2 - 1) % L, j2]) + int(lat[(i2 + 1) % L, j2]) +
              int(lat[i2, (j2 - 1) % L]) + int(lat[i2, (j2 + 1) % L]) -
              n1)

        return self.J * (n1 - n2) * (s1 - s2)

    def step(self) -> int:
        """
        Perform one Monte Carlo sweep (L*L attempted swaps).

        Returns
        -------
        n_accepted : number of accepted swaps in this sweep
        """
        L = self.L
        lat = self.lattice
        rng = self.rng
        beta = self.beta
        n_accepted = 0

        for _ in range(L * L):
            # Random site
            i1 = int(rng.integers(L))
            j1 = int(rng.integers(L))

            # Random nearest neighbour (0=up, 1=down, 2=left, 3=right)
            d = int(rng.integers(4))
            if d == 0:
                i2, j2 = (i1 - 1) % L, j1
            elif d == 1:
                i2, j2 = (i1 + 1) % L, j1
            elif d == 2:
                i2, j2 = i1, (j1 - 1) % L
            else:
                i2, j2 = i1, (j1 + 1) % L

            # Skip if both sites are the same type
            if lat[i1, j1] == lat[i2, j2]:
                continue

            dE = self._delta_energy(i1, j1, i2, j2)

            if dE <= 0 or rng.random() < np.exp(-beta * dE):
                lat[i1, j1], lat[i2, j2] = lat[i2, j2], lat[i1, j1]
                n_accepted += 1

        self.sweep += 1
        return n_accepted

    # ------------------------------------------------------------------
    # High-level run
    # ------------------------------------------------------------------

    def run(self, n_sweeps: int, measure_every: int = 1) -> dict:
        """
        Run the simulation for *n_sweeps* Monte Carlo sweeps.

        Parameters
        ----------
        n_sweeps      : total number of sweeps to perform
        measure_every : record observables every this many sweeps

        Returns
        -------
        dict with keys:
            'sweep'        : array of sweep indices at which measurements were taken
            'energy'       : total energy at each measurement
            'energy_per_site' : energy per lattice site
            'acceptance'   : acceptance rate at each measurement sweep
        """
        sweeps, energies, acceptance = [], [], []

        for _ in range(n_sweeps):
            n_acc = self.step()
            if self.sweep % measure_every == 0:
                sweeps.append(self.sweep)
                energies.append(self.energy())
                # acceptance rate: n_acc out of L*L attempts where sites differ
                acceptance.append(n_acc / (self.L * self.L))

        energies = np.array(energies)
        return {
            "sweep": np.array(sweeps),
            "energy": energies,
            "energy_per_site": energies / (self.L * self.L),
            "acceptance": np.array(acceptance),
        }
