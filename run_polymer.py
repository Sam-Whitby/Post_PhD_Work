"""
Run the lattice-polymer Kawasaki simulation and visualise it.

Usage
-----
    python run_polymer.py

Produces:
  1. Coupling-matrix panel  – shows J[k1,k2] for monomer pairs at
     each of the four possible initial separations (0, 1, √2, >√2).
  2. Combined animation     – left: lattice with backbone path overlaid,
                              right: live energy-vs-sweep plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from polymer_kawasaki import (
    PolymerKawasaki,
    backbone_coupling_matrix,
    generate_moore_curve,
    initial_distance_matrix,
)


# ── Parameters ────────────────────────────────────────────────────────────────
L               = 8      # lattice side (must be power of 2)
J_BACKBONE      = 1.0    # coupling strength along the backbone
T               = 1.0    # temperature in units of J/k_B
SEED            = 42

N_FRAMES        = 300    # animation frames
SWEEPS_PER_FRAME = 2     # MC sweeps between frames
# ─────────────────────────────────────────────────────────────────────────────


# ── Coupling-matrix visualisation ─────────────────────────────────────────────

def plot_coupling_matrices(J_matrix: np.ndarray,
                           moore_coords: list) -> None:
    """
    Show the N×N coupling matrix J[k1,k2] split into four panels, one for
    each initial inter-monomer distance category:
        d = 0       : self (diagonal)
        d = 1       : orthogonal lattice neighbours
        d = √2      : diagonal lattice neighbours
        d > √2      : further apart
    """
    D = initial_distance_matrix(moore_coords)
    sqrt2 = np.sqrt(2)
    tol   = 1e-9

    masks = {
        r"$d = 0$":      D < tol,
        r"$d = 1$":      np.abs(D - 1.0)    < tol,
        r"$d = \sqrt{2}$": np.abs(D - sqrt2) < tol,
        r"$d > \sqrt{2}$": D > sqrt2 + tol,
    }

    vmax = np.max(np.abs(J_matrix)) or 1.0
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        "Coupling matrix $J[k_1,k_2]$ by initial monomer separation\n"
        "(white = outside this distance category)",
        fontsize=11,
    )

    for ax, (label, mask) in zip(axes, masks.items()):
        J_masked = np.where(mask, J_matrix, np.nan)
        im = ax.imshow(
            J_masked, cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            interpolation="nearest", aspect="auto",
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Monomer $k_2$", fontsize=9)
        ax.set_ylabel("Monomer $k_1$", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, shrink=0.75, label="$J$")

    plt.tight_layout()
    plt.savefig("coupling_matrices.png", dpi=150)
    print("Saved coupling_matrices.png")
    plt.show()


# ── Combined animation ────────────────────────────────────────────────────────

def animate_combined(sim: PolymerKawasaki,
                     n_frames: int = N_FRAMES,
                     sweeps_per_frame: int = SWEEPS_PER_FRAME) -> animation.FuncAnimation:
    """
    Single-figure animation:
      Left  – lattice coloured by monomer index with backbone path overlaid.
      Right – energy per site vs Monte Carlo sweep, growing in real time.
    """
    N = sim.N
    L = sim.L

    sweep_history  = [sim.sweep]
    energy_history = [sim.energy() / N]

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5.5))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.97, wspace=0.35)

    ax_lat = fig.add_subplot(1, 2, 1)
    ax_en  = fig.add_subplot(1, 2, 2)

    # ── Lattice panel ─────────────────────────────────────────────────────
    img = ax_lat.imshow(
        sim.lattice, cmap="viridis",
        vmin=0, vmax=N - 1,
        interpolation="nearest", origin="upper",
    )
    ax_lat.set_title("Lattice  (colour = monomer index)", fontsize=11, pad=6)
    ax_lat.set_xticks(np.arange(L))
    ax_lat.set_yticks(np.arange(L))
    ax_lat.tick_params(labelsize=7)
    plt.colorbar(img, ax=ax_lat, label="Monomer index", shrink=0.75)

    # Backbone path as a LineCollection coloured by position along chain
    rows, cols = sim.get_backbone_coords()
    pts = np.column_stack([cols.astype(float), rows.astype(float)])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = plt.cm.plasma(np.linspace(0, 1, N - 1))
    lc = LineCollection(segments, colors=seg_colors, linewidths=1.8, zorder=2)
    ax_lat.add_collection(lc)

    # Sweep counter below the lattice
    sweep_text = ax_lat.text(
        0.5, -0.06, f"Sweep: {sim.sweep}",
        transform=ax_lat.transAxes,
        ha="center", va="top", fontsize=10,
    )

    # ── Energy panel ──────────────────────────────────────────────────────
    ax_en.set_xlabel("Monte Carlo sweep", fontsize=10)
    ax_en.set_ylabel("Energy per site  $E/N$", fontsize=10)
    ax_en.set_title("Energy vs time", fontsize=11, pad=6)
    ax_en.tick_params(labelsize=9)
    (line,) = ax_en.plot(sweep_history, energy_history, lw=1.2, color="steelblue")

    # ── Update function ───────────────────────────────────────────────────
    def update(_frame):
        for _ in range(sweeps_per_frame):
            sim.step()

        sweep_history.append(sim.sweep)
        energy_history.append(sim.energy() / N)

        # Update lattice image
        img.set_data(sim.lattice)

        # Update backbone path
        r, c = sim.get_backbone_coords()
        new_pts  = np.column_stack([c.astype(float), r.astype(float)])
        new_segs = np.stack([new_pts[:-1], new_pts[1:]], axis=1)
        lc.set_segments(new_segs)

        sweep_text.set_text(f"Sweep: {sim.sweep}")

        # Update energy plot with dynamic axes
        line.set_xdata(sweep_history)
        line.set_ydata(energy_history)
        ax_en.set_xlim(sweep_history[0], sweep_history[-1] + 1)
        ymin, ymax = min(energy_history), max(energy_history)
        margin = abs(ymax - ymin) * 0.1 if ymax != ymin else 0.5
        ax_en.set_ylim(ymin - margin, ymax + margin)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=60, blit=False
    )
    plt.show()
    return ani


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    N = L * L
    print(f"Lattice: {L}×{L}  ({N} monomers)")
    print(f"T = {T},  J_backbone = {J_BACKBONE}")

    # Build coupling matrix and initial configuration
    J_matrix    = backbone_coupling_matrix(N, J=J_BACKBONE)
    moore_coords = generate_moore_curve(order=round(np.log2(L)))

    print(f"\nMoore curve spans rows 0–{max(r for r,c in moore_coords)}, "
          f"cols 0–{max(c for r,c in moore_coords)}")
    print(f"All {N} sites visited: "
          f"{'YES' if len(set(moore_coords)) == N else 'NO'}")

    # ── Coupling matrix visualisation ─────────────────────────────────────
    print("\nPlotting coupling matrices by initial separation …")
    plot_coupling_matrices(J_matrix, moore_coords)

    # ── Run simulation ────────────────────────────────────────────────────
    sim = PolymerKawasaki(L=L, J_matrix=J_matrix, T=T, seed=SEED, init="moore")

    E0 = sim.energy()
    print(f"\nInitial energy: {E0:.2f}  (per site: {E0/N:.4f})")
    print(f"Maximum possible (all backbone bonds adjacent): "
          f"{-(N-1)*J_BACKBONE:.2f}")

    print("\nStarting animation …  (close the window to exit)")
    ani = animate_combined(sim)   # keep reference – prevents GC


if __name__ == "__main__":
    main()
