"""
Kawasaki dynamics for a fully-occupied 2D lattice polymer.

The lattice (L×L) is completely filled: every site holds exactly one monomer,
labelled 0 … N-1 (N = L²).  An arbitrary N×N coupling matrix J_matrix[k1, k2]
defines the interaction between monomers k1 and k2 when they occupy adjacent
lattice sites.

Hamiltonian:
    H = -sum_{<r, r'>} J_matrix[sigma(r), sigma(r')]

where <r,r'> denotes a nearest-neighbour pair (Moore / 8-neighbour) and
sigma(r) is the monomer index at site r.

Kawasaki move: swap two neighbouring monomers.  The identity of each monomer
is conserved; only their spatial positions change.

Typical use: backbone-only coupling  J_matrix[k, k±1] = J (all other entries
zero).  Then H is minimised when as many consecutive backbone monomers as
possible are spatially adjacent – i.e. the polymer is in a compact, path-
filling conformation.  Starting from a Moore space-filling curve (which
achieves this minimum) lets you study how the polymer unfolds with temperature.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Moore curve
# ──────────────────────────────────────────────────────────────────────────────

def generate_moore_curve(order: int) -> list[tuple[int, int]]:
    """
    Return the ordered list of (row, col) sites visited by a Moore space-
    filling curve of the given order, covering a 2^order × 2^order grid.

    The Moore curve is generated via its L-system:
        Axiom : LFL+F+LFL
        L     → -RF+LFL+FR-
        R     → +LF-RFR-FL+
        F = step forward, + = turn left 90°, - = turn right 90°

    Applying the rules (order-1) times and then interpreting the string with a
    turtle gives 2^(2·order) distinct lattice sites.
    """
    axiom = "LFL+F+LFL"
    rules = {"L": "-RF+LFL+FR-", "R": "+LF-RFR-FL+"}

    s = axiom
    for _ in range(order - 1):
        s = "".join(rules.get(c, c) for c in s)

    # Turtle: directions in (row, col) – East, North, West, South
    dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    r, c, d = 0, 0, 0
    coords: list[tuple[int, int]] = [(r, c)]

    for cmd in s:
        if cmd == "F":
            r += dirs[d][0]
            c += dirs[d][1]
            coords.append((r, c))
        elif cmd == "+":      # turn left (CCW)
            d = (d + 1) % 4
        elif cmd == "-":      # turn right (CW)
            d = (d - 1) % 4

    # Shift so the top-left corner is (0, 0)
    min_r = min(x[0] for x in coords)
    min_c = min(x[1] for x in coords)
    coords = [(x[0] - min_r, x[1] - min_c) for x in coords]

    assert len(coords) == 2 ** (2 * order), (
        f"Moore curve gave {len(coords)} points, expected {2**(2*order)}"
    )
    assert len(set(coords)) == len(coords), "Moore curve has repeated sites"
    return coords


# ──────────────────────────────────────────────────────────────────────────────
# Coupling matrices
# ──────────────────────────────────────────────────────────────────────────────

def backbone_coupling_matrix(N: int, J: float = 1.0) -> np.ndarray:
    """
    Return an N×N coupling matrix where only adjacent backbone pairs interact:
        J_matrix[k, k±1] = J   (all other entries zero).

    This is the simplest polymer model: only monomers that are directly bonded
    along the chain attract each other when they are spatially adjacent.
    """
    J_mat = np.zeros((N, N))
    for k in range(N - 1):
        J_mat[k, k + 1] = J
        J_mat[k + 1, k] = J
    return J_mat


def initial_distance_matrix(moore_coords: list[tuple[int, int]]) -> np.ndarray:
    """
    Return the N×N matrix of Euclidean distances between all monomer pairs
    in the initial (Moore curve) configuration.

    D[k1, k2] = ||r_{k1} - r_{k2}||_2   at t = 0.
    """
    coords = np.array(moore_coords, dtype=float)          # (N, 2)
    diff = coords[:, None, :] - coords[None, :, :]        # (N, N, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))             # (N, N)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation class
# ──────────────────────────────────────────────────────────────────────────────

class PolymerKawasaki:
    """
    Kawasaki dynamics for a fully-occupied lattice polymer.

    Attributes
    ----------
    lattice : (L, L) int array
        lattice[i, j]  = index of the monomer currently at site (i, j).
    sweep   : int
        Number of completed Monte Carlo sweeps.
    """

    def __init__(
        self,
        L: int,
        J_matrix: np.ndarray,
        T: float = 1.0,
        seed: Optional[int] = None,
        init: str = "moore",
        enforce_backbone: bool = False,
    ):
        """
        Parameters
        ----------
        L                : lattice side length (must be a power of 2 for init='moore')
        J_matrix         : N×N coupling matrix  (N = L²)
        T                : temperature in units of J / k_B
        seed             : RNG seed for reproducibility
        init             : 'moore'  – place monomers along the Moore space-filling curve
                           'random' – random permutation of monomer labels
        enforce_backbone : if True, hard-reject any swap that would place consecutive
                           backbone monomers (k, k+1) more than √2 apart.  This acts
                           as an infinite repulsion for stretched bonds, keeping the
                           polymer chain intact at all times.
        """
        N = L * L
        if J_matrix.shape != (N, N):
            raise ValueError(f"J_matrix must be {N}×{N}, got {J_matrix.shape}")
        if init not in ("moore", "random"):
            raise ValueError("init must be 'moore' or 'random'")

        self.L = L
        self.N = N
        self.J = J_matrix.copy()
        self.T = T
        self.beta = 1.0 / T
        self.rng = np.random.default_rng(seed)
        self.sweep = 0
        self.enforce_backbone = enforce_backbone

        # lattice[i, j] = monomer index
        self.lattice = np.empty((L, L), dtype=int)

        if init == "moore":
            order = round(math.log2(L))
            if 2 ** order != L:
                raise ValueError("L must be a power of 2 for init='moore'")
            coords = generate_moore_curve(order)
            for k, (r, c) in enumerate(coords):
                self.lattice[r, c] = k
        else:
            perm = self.rng.permutation(N)
            self.lattice = perm.reshape(L, L)

        # Inverse position map: _pos[k] = (row, col) of monomer k.
        # Maintained incrementally in step() for O(1) backbone-bond checks.
        self._pos = np.empty((N, 2), dtype=int)
        self._pos[self.lattice.ravel(), 0] = np.repeat(np.arange(L), L)
        self._pos[self.lattice.ravel(), 1] = np.tile(np.arange(L), L)

    # ── Observables ────────────────────────────────────────────────────────

    def energy(self) -> float:
        """Total energy – vectorised, sums over all 8-neighbour bond directions."""
        lat = self.lattice
        e = 0.0
        for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            lat_shift = np.roll(np.roll(lat, -di, axis=0), -dj, axis=1)
            k1 = lat.ravel()
            k2 = lat_shift.ravel()
            e -= float(np.sum(self.J[k1, k2]))
        return e

    def get_backbone_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (rows, cols) arrays of length N such that
        rows[k], cols[k] is the current lattice position of monomer k.
        """
        return self._pos[:, 0].copy(), self._pos[:, 1].copy()

    # ── Backbone constraint ────────────────────────────────────────────────

    def _backbone_swap_valid(self, i1: int, j1: int, i2: int, j2: int,
                              a: int, b: int) -> bool:
        """
        Return True iff swapping monomers a (at i1,j1) ↔ b (at i2,j2) keeps
        every backbone bond (k, k+1) within distance √2 (i.e. dist² ≤ 2).

        Only bonds touching a or b can change.  The bond between a and b
        itself is unaffected (the two monomers simply exchange positions so
        the distance is unchanged).

        Distances use the minimum-image convention for periodic boundaries.
        """
        L = self.L
        N = self.N
        pos = self._pos

        def row(k: int) -> int:
            return i2 if k == a else (i1 if k == b else int(pos[k, 0]))

        def col(k: int) -> int:
            return j2 if k == a else (j1 if k == b else int(pos[k, 1]))

        def bond_ok(k1: int, k2: int) -> bool:
            dr = abs(row(k1) - row(k2)); dr = min(dr, L - dr)
            dc = abs(col(k1) - col(k2)); dc = min(dc, L - dc)
            return dr * dr + dc * dc <= 2   # ≤ (√2)²

        if a > 0     and not bond_ok(a - 1, a):  return False
        if a < N - 1 and not bond_ok(a, a + 1):  return False
        if b > 0     and not bond_ok(b - 1, b):  return False
        if b < N - 1 and not bond_ok(b, b + 1):  return False
        return True

    # ── Monte Carlo ────────────────────────────────────────────────────────

    def _delta_energy(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """
        Exact energy change for swapping monomers at (i1,j1) and (i2,j2).

        With H = -Σ J[σ(r),σ(r')], swapping monomers a ↔ b gives:

            ΔE = Σ_{r'≠(i2,j2), adj (i1,j1)} [J(a,c) - J(b,c)]
               + Σ_{r'≠(i1,j1), adj (i2,j2)} [J(b,c) - J(a,c)]

        where c = σ(r') and J is symmetric.
        """
        L = self.L
        lat = self.lattice
        a = int(lat[i1, j1])
        b = int(lat[i2, j2])

        dE = 0.0
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   ( 0, -1),          ( 0, 1),
                   ( 1, -1), ( 1, 0), ( 1, 1)]

        for di, dj in offsets:
            ni, nj = (i1 + di) % L, (j1 + dj) % L
            if ni == i2 and nj == j2:
                continue
            c = int(lat[ni, nj])
            dE += self.J[a, c] - self.J[b, c]

        for di, dj in offsets:
            ni, nj = (i2 + di) % L, (j2 + dj) % L
            if ni == i1 and nj == j1:
                continue
            c = int(lat[ni, nj])
            dE += self.J[b, c] - self.J[a, c]

        return dE

    def step(self) -> int:
        """
        One Monte Carlo sweep: L² attempted nearest-neighbour swaps.
        Returns the number of accepted moves.
        """
        L = self.L
        lat = self.lattice
        rng = self.rng
        beta = self.beta
        n_accepted = 0

        check_bb = self.enforce_backbone
        pos = self._pos

        for _ in range(L * L):
            i1 = int(rng.integers(L))
            j1 = int(rng.integers(L))

            d = int(rng.integers(8))
            di = (-1,  1, 0,  0, -1, -1,  1,  1)[d]
            dj = ( 0,  0,-1,  1, -1,  1, -1,  1)[d]
            i2, j2 = (i1 + di) % L, (j1 + dj) % L

            a = int(lat[i1, j1])
            b = int(lat[i2, j2])

            # Hard backbone constraint: reject if any bond would stretch > √2
            if check_bb and not self._backbone_swap_valid(i1, j1, i2, j2, a, b):
                continue

            dE = self._delta_energy(i1, j1, i2, j2)

            if dE <= 0 or rng.random() < np.exp(-beta * dE):
                lat[i1, j1], lat[i2, j2] = lat[i2, j2], lat[i1, j1]
                pos[a, 0], pos[a, 1] = i2, j2
                pos[b, 0], pos[b, 1] = i1, j1
                n_accepted += 1

        self.sweep += 1
        return n_accepted

    def run(self, n_sweeps: int, measure_every: int = 1) -> dict:
        """Run for n_sweeps MC sweeps, measuring every measure_every sweeps."""
        sweeps, energies, acceptance = [], [], []
        for _ in range(n_sweeps):
            n_acc = self.step()
            if self.sweep % measure_every == 0:
                sweeps.append(self.sweep)
                energies.append(self.energy())
                acceptance.append(n_acc / self.N)
        energies = np.array(energies)
        return {
            "sweep":           np.array(sweeps),
            "energy":          energies,
            "energy_per_site": energies / self.N,
            "acceptance":      np.array(acceptance),
        }
