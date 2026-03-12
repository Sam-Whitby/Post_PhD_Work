"""
Kawasaki dynamics simulation of a 2D lattice gas.

In Kawasaki dynamics the total number of particles is conserved. At each
Monte Carlo step a random nearest-neighbour pair is chosen; if the two sites
differ (one occupied, one empty) a swap is attempted and accepted via the
Metropolis criterion.

Hamiltonian:  H = -J * sum_{<i,j>} n_i * n_j
where n_i = 1 (occupied) or 0 (empty) and the sum runs over all 8 neighbours
(Moore neighbourhood) on a periodic square lattice.

Positive J favours clustering (phase separation at low T).
"""

from typing import Optional

import numpy as np


class KawasakiLatticeGas:
    """2D lattice gas with Kawasaki (conserved) dynamics."""

    def __init__(self, L: int, density: float, J: float = 1.0, T: float = 2.0,
                 seed: Optional[int] = None, init: str = "circle"):
        """
        Parameters
        ----------
        L       : linear size of the square lattice (L x L sites)
        density : fraction of sites that are occupied  (0 < density < 1)
        J       : coupling constant  (J > 0  → attractive interactions)
        T       : temperature in units of J/k_B
        seed    : random seed for reproducibility
        init    : initial condition — 'circle' (condensed disk at centre)
                  or 'random' (uniform random)
        """
        if not 0 < density < 1:
            raise ValueError("density must be between 0 and 1 (exclusive)")
        if init not in ("circle", "random"):
            raise ValueError("init must be 'circle' or 'random'")

        self.L = L
        self.J = J
        self.T = T
        self.beta = 1.0 / T if T > 0.0 else np.inf

        self.rng = np.random.default_rng(seed)

        n_particles = round(density * L * L)

        if init == "circle":
            # Fill the n_particles sites closest to the lattice centre,
            # giving a compact disk as the starting configuration.
            cx, cy = (L - 1) / 2.0, (L - 1) / 2.0
            ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
            dist2 = (ii - cx) ** 2 + (jj - cy) ** 2
            order = np.argsort(dist2.ravel())
            flat = np.zeros(L * L, dtype=np.int8)
            flat[order[:n_particles]] = 1
            self.lattice = flat.reshape(L, L)
        else:
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
        """Total lattice energy (vectorised, no double-counting).

        Sums over the 4 unique bond directions that tile the 8-neighbour
        (Moore) lattice without double-counting:
          axis-0 shift  → vertical bonds
          axis-1 shift  → horizontal bonds
          (+1,+1) shift → one diagonal family
          (+1,-1) shift → other diagonal family
        """
        lat = self.lattice
        e = -self.J * (
            np.sum(lat * np.roll(lat, 1, axis=0)) +                          # vertical
            np.sum(lat * np.roll(lat, 1, axis=1)) +                          # horizontal
            np.sum(lat * np.roll(np.roll(lat, 1, axis=0), 1, axis=1)) +     # diagonal ↘
            np.sum(lat * np.roll(np.roll(lat, 1, axis=0), -1, axis=1))      # diagonal ↙
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

        # All 8 Moore neighbours of site 1, then subtract site 2's contribution
        s1 = (int(lat[(i1 - 1) % L, (j1 - 1) % L]) +
              int(lat[(i1 - 1) % L,  j1             ]) +
              int(lat[(i1 - 1) % L, (j1 + 1) % L]) +
              int(lat[ i1,           (j1 - 1) % L]) +
              int(lat[ i1,           (j1 + 1) % L]) +
              int(lat[(i1 + 1) % L, (j1 - 1) % L]) +
              int(lat[(i1 + 1) % L,  j1             ]) +
              int(lat[(i1 + 1) % L, (j1 + 1) % L]) -
              n2)

        # All 8 Moore neighbours of site 2, then subtract site 1's contribution
        s2 = (int(lat[(i2 - 1) % L, (j2 - 1) % L]) +
              int(lat[(i2 - 1) % L,  j2             ]) +
              int(lat[(i2 - 1) % L, (j2 + 1) % L]) +
              int(lat[ i2,           (j2 - 1) % L]) +
              int(lat[ i2,           (j2 + 1) % L]) +
              int(lat[(i2 + 1) % L, (j2 - 1) % L]) +
              int(lat[(i2 + 1) % L,  j2             ]) +
              int(lat[(i2 + 1) % L, (j2 + 1) % L]) -
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

            # Random neighbour from all 8 Moore directions
            # (di, dj) offsets: N, S, W, E, NW, NE, SW, SE
            d = int(rng.integers(8))
            di = (-1, 1, 0, 0, -1, -1, 1,  1)[d]
            dj = ( 0, 0,-1, 1, -1,  1,-1,  1)[d]
            i2, j2 = (i1 + di) % L, (j1 + dj) % L

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
