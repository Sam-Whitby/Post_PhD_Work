"""
Run the lattice-polymer Kawasaki simulation and visualise it.

Usage
-----
    python run_polymer.py [options]

    --L              INT    lattice side length, must be power of 2 (default: 8)
    --J              FLOAT  backbone coupling strength (default: 1.0)
    --E1             FLOAT  attraction for non-backbone orthogonal contacts (default: 0.0)
    --E2             FLOAT  attraction for non-backbone diagonal contacts (default: 0.0)
    --T              FLOAT  temperature in units of k_B (default: 1.0)
    --K              FLOAT  harmonic spring constant for backbone bonds (default: 10.0)
    --seed           INT    random seed (default: 42)
    --frames         INT    number of animation frames (default: 300)
    --sweeps-per-frame INT  MC sweeps between frames (default: 2)

Examples
--------
    python run_polymer.py --T 2.0 --K 5.0
    python run_polymer.py --L 16 --K 0 --T 0.5
    python run_polymer.py --E1 0.5 --E2 0.25 --K 10

Produces:
  1. Coupling-matrix panel  – shows J[k1,k2] for monomer pairs at
     each of the four possible initial separations (0, 1, √2, >√2).
  2. Combined animation     – left: lattice with backbone path overlaid,
                              right: live energy-vs-sweep plot.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from polymer_kawasaki import (
    PolymerKawasaki,
    backbone_coupling_matrix,
    moore_contact_matrix,
    generate_moore_curve,
    initial_distance_matrix,
)


# ── Parameters ────────────────────────────────────────────────────────────────
L                = 8      # lattice side (must be power of 2)
J_BACKBONE       = 0.0    # backbone bond coupling (>0 adds extra attraction when backbone monomers are adjacent)
E1               = 0.0    # attraction for non-backbone orthogonal contacts (dist=1)
E2               = 0.0    # attraction for non-backbone diagonal contacts  (dist=√2)
T                = 1.0    # temperature in units of k_B
SEED             = 42
K                = 10.0   # harmonic spring constant for backbone bonds (V = K * dist²)
                           # large K → chain stays compact; K = 0 → no bond penalty

N_FRAMES         = 300    # animation frames
SWEEPS_PER_FRAME = 2      # MC sweeps between frames
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

    # NaN is used for "outside this distance category" – map it to white so
    # it is indistinguishable from J=0 and the plot shows exactly two colours.
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        "Coupling matrix $J[k_1,k_2]$ by initial monomer separation",
        fontsize=11,
    )

    for ax, (label, mask) in zip(axes, masks.items()):
        J_masked = np.where(mask, J_matrix, np.nan)
        im = ax.imshow(
            J_masked, cmap=cmap,
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


# ── Periodic bond segments ────────────────────────────────────────────────────

def _bond_segments_periodic(rows: np.ndarray, cols: np.ndarray,
                             L: int) -> np.ndarray:
    """
    Build a (2*(N-1), 2, 2) segment array for backbone bonds on a periodic
    lattice, handling boundary wrapping correctly.

    For each bond k → k+1:
    - If it does NOT cross a boundary: one normal segment + one degenerate
      (zero-length) placeholder.
    - If it DOES cross a boundary: two half-segments, each exiting at the
      boundary midpoint (0.5 units past the edge), so no line spans the
      full plot width/height.

    The total segment count is always 2*(N-1), keeping set_segments() happy.
    Axes convention: x = col, y = row.
    """
    N = len(rows)
    segs = np.empty((2 * (N - 1), 2, 2), dtype=float)

    for k in range(N - 1):
        x1, y1 = float(cols[k]),     float(rows[k])
        x2, y2 = float(cols[k + 1]), float(rows[k + 1])

        # Minimum-image offset
        dx = x2 - x1
        if   dx >  L / 2: dx -= L
        elif dx < -L / 2: dx += L

        dy = y2 - y1
        if   dy >  L / 2: dy -= L
        elif dy < -L / 2: dy += L

        wraps = (dx != x2 - x1) or (dy != y2 - y1)

        if not wraps:
            segs[2 * k]     = [(x1, y1), (x2, y2)]
            segs[2 * k + 1] = [(x1, y1), (x1, y1)]   # degenerate placeholder
        else:
            # Each endpoint gets a half-segment exiting 0.5 units past its edge
            segs[2 * k]     = [(x1, y1), (x1 + 0.5 * dx, y1 + 0.5 * dy)]
            segs[2 * k + 1] = [(x2, y2), (x2 - 0.5 * dx, y2 - 0.5 * dy)]

    return segs


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
    ax_lat.set_facecolor("#f0f0f0")
    ax_lat.set_xlim(-0.6, L - 0.4)
    ax_lat.set_ylim(L - 0.4, -0.6)   # y increases downward to match row indexing
    ax_lat.set_aspect("equal")
    ax_lat.set_xticks(np.arange(L))
    ax_lat.set_yticks(np.arange(L))
    ax_lat.tick_params(labelsize=7)
    ax_lat.set_title("Polymer backbone  (colour = monomer index)", fontsize=11, pad=6)

    # Backbone bonds – periodic-boundary-aware segments
    # Each bond k→k+1 produces 2 entries (see _bond_segments_periodic).
    # Colours are duplicated to match.
    rows, cols = sim.get_backbone_coords()
    bond_base_colors = plt.cm.plasma(np.linspace(0, 1, N - 1))
    seg_colors = np.repeat(bond_base_colors, 2, axis=0)   # shape (2*(N-1), 4)
    segments = _bond_segments_periodic(rows, cols, L)
    lc = LineCollection(segments, colors=seg_colors, linewidths=2.5, zorder=2)
    ax_lat.add_collection(lc)

    # Monomer circles: position fixed at grid sites, colour = monomer index
    site_rows = np.repeat(np.arange(L), L).astype(float)
    site_cols = np.tile(np.arange(L), L).astype(float)
    sc = ax_lat.scatter(
        site_cols, site_rows,
        c=sim.lattice.ravel(), cmap="plasma",
        vmin=0, vmax=N - 1,
        s=260, zorder=3, linewidths=0.6, edgecolors="k",
    )
    plt.colorbar(sc, ax=ax_lat, label="Monomer index", shrink=0.75)

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

        # Update monomer circle colours
        sc.set_array(sim.lattice.ravel().astype(float))

        # Update backbone bond segments (periodic wrapping handled)
        r, c = sim.get_backbone_coords()
        lc.set_segments(_bond_segments_periodic(r, c, L))

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
    parser = argparse.ArgumentParser(description="Lattice-polymer Kawasaki simulation")
    parser.add_argument("--L",                type=int,   default=L,                help="lattice side length (power of 2)")
    parser.add_argument("--J",                type=float, default=J_BACKBONE,        help="backbone coupling strength")
    parser.add_argument("--E1",               type=float, default=E1,                help="non-backbone orthogonal contact strength (dist=1)")
    parser.add_argument("--E2",               type=float, default=E2,                help="non-backbone diagonal contact strength (dist=√2)")
    parser.add_argument("--T",                type=float, default=T,                 help="temperature")
    parser.add_argument("--K",                type=float, default=K,                 help="harmonic spring constant")
    parser.add_argument("--seed",             type=int,   default=SEED,              help="random seed")
    parser.add_argument("--frames",           type=int,   default=N_FRAMES,          help="animation frames")
    parser.add_argument("--sweeps-per-frame", type=int,   default=SWEEPS_PER_FRAME,  help="MC sweeps between frames")
    args = parser.parse_args()

    L_       = args.L
    J_val    = args.J
    E1_      = args.E1
    E2_      = args.E2
    T_       = args.T
    K_       = args.K
    seed_    = args.seed
    frames_  = args.frames
    spf_     = args.sweeps_per_frame

    N = L_ * L_
    print(f"Lattice: {L_}×{L_}  ({N} monomers)")
    print(f"T = {T_},  J = {J_val},  E1 = {E1_},  E2 = {E2_},  K = {K_}")

    moore_coords = generate_moore_curve(order=round(np.log2(L_)))
    J_matrix = moore_contact_matrix(moore_coords, J=J_val, E1=E1_, E2=E2_)

    print(f"\nMoore curve spans rows 0–{max(r for r,c in moore_coords)}, "
          f"cols 0–{max(c for r,c in moore_coords)}")
    print(f"All {N} sites visited: "
          f"{'YES' if len(set(moore_coords)) == N else 'NO'}")

    # ── Coupling matrix visualisation ─────────────────────────────────────
    print("\nPlotting coupling matrices by initial separation …")
    plot_coupling_matrices(J_matrix, moore_coords)

    # ── Run simulation ────────────────────────────────────────────────────
    sim = PolymerKawasaki(L=L_, J_matrix=J_matrix, T=T_, seed=seed_,
                          init="moore", K=K_)

    E0 = sim.energy()
    print(f"\nInitial energy: {E0:.2f}")

    print("\nStarting animation …  (close the window to exit)")
    ani = animate_combined(sim, n_frames=frames_, sweeps_per_frame=spf_)


if __name__ == "__main__":
    main()
