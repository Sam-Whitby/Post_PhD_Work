"""
Run a Kawasaki lattice-gas simulation and visualise the results.

Usage
-----
    python run_kawasaki.py [options]

    --L               INT    lattice size L×L (default: 64)
    --density         FLOAT  fraction of occupied sites (default: 0.4)
    --J               FLOAT  coupling constant (default: 1.0)
    --T               FLOAT  temperature (default: 1.5)
    --seed            INT    random seed (default: 42)
    --frames          INT    number of animation frames (default: 300)
    --sweeps-per-frame INT   MC sweeps between frames (default: 3)

Examples
--------
    python run_kawasaki.py --T 2.5 --density 0.3
    python run_kawasaki.py --L 32 --J 2.0 --T 1.0

Produces:
  - A combined animation: lattice image on the left, live energy plot on the right
  - A static plot of the structure factor after the run
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kawasaki_lattice_gas import KawasakiLatticeGas


# ── Simulation parameters ──────────────────────────────────────────────────────
L               = 64    # lattice size  (L x L)
DENSITY         = 0.4   # fraction of occupied sites
J               = 1.0   # coupling constant  (J > 0 → attraction → phase separation)
T               = 1.5   # temperature  (critical T ≈ 2.27 J/k_B for 2-D Ising)
SEED            = 42

N_FRAMES        = 300   # number of animation frames
SWEEPS_PER_FRAME = 3    # MC sweeps between frames
# ──────────────────────────────────────────────────────────────────────────────


def animate_combined(sim: KawasakiLatticeGas,
                     n_frames: int = N_FRAMES,
                     sweeps_per_frame: int = SWEEPS_PER_FRAME):
    """
    Single figure animation with:
      - Left  : lattice occupation (black = particle, white = empty)
      - Right : energy per site vs Monte Carlo sweep, growing in real time
    """
    # Seed the energy history with the initial state
    sweep_history = [sim.sweep]
    energy_history = [sim.energy() / sim.L ** 2]

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.07, right=0.97, wspace=0.35)

    ax_lat = fig.add_subplot(1, 2, 1)
    ax_en  = fig.add_subplot(1, 2, 2)

    # Lattice panel
    ax_lat.set_axis_off()
    img = ax_lat.imshow(sim.lattice, cmap="binary", vmin=0, vmax=1,
                        interpolation="nearest")
    ax_lat.set_title("Lattice", fontsize=11, pad=6)

    # Sweep counter as a text box below the lattice panel
    sweep_text = ax_lat.text(
        0.5, -0.03, f"Sweep: {sim.sweep}",
        transform=ax_lat.transAxes,
        ha="center", va="top", fontsize=10,
    )

    # Energy panel
    ax_en.set_xlabel("Monte Carlo sweep", fontsize=10)
    ax_en.set_ylabel("Energy per site  $E/N$", fontsize=10)
    ax_en.set_title("Energy vs time", fontsize=11, pad=6)
    ax_en.tick_params(labelsize=9)
    (line,) = ax_en.plot(sweep_history, energy_history, lw=1.2, color="steelblue")

    # ── Update function ────────────────────────────────────────────────
    def update(frame):
        for _ in range(sweeps_per_frame):
            sim.step()

        sweep_history.append(sim.sweep)
        energy_history.append(sim.energy() / sim.L ** 2)

        img.set_data(sim.lattice)
        sweep_text.set_text(f"Sweep: {sim.sweep}")

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


def plot_structure_factor(sim: KawasakiLatticeGas) -> None:
    sf = np.fft.fftshift(sim.structure_factor())
    sf[sim.L // 2, sim.L // 2] = 0   # remove zero-mode (total density)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log1p(sf), origin="lower", cmap="inferno",
                   extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar(im, ax=ax, label=r"$\ln(1 + S(\mathbf{q}))$")
    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    ax.set_title("Static structure factor")
    plt.tight_layout()
    plt.savefig("structure_factor.png", dpi=150)
    print("Saved structure_factor.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Kawasaki lattice-gas simulation")
    parser.add_argument("--L",                type=int,   default=L,               help="lattice size")
    parser.add_argument("--density",          type=float, default=DENSITY,          help="fraction of occupied sites")
    parser.add_argument("--J",                type=float, default=J,                help="coupling constant")
    parser.add_argument("--T",                type=float, default=T,                help="temperature")
    parser.add_argument("--seed",             type=int,   default=SEED,             help="random seed")
    parser.add_argument("--frames",           type=int,   default=N_FRAMES,         help="animation frames")
    parser.add_argument("--sweeps-per-frame", type=int,   default=SWEEPS_PER_FRAME, help="MC sweeps between frames")
    args = parser.parse_args()

    print(f"Initialising {args.L}×{args.L} lattice gas  "
          f"(density={args.density}, J={args.J}, T={args.T}, init=circle)")

    sim = KawasakiLatticeGas(L=args.L, density=args.density, J=args.J, T=args.T,
                             seed=args.seed, init="circle")

    print("Showing combined animation (lattice + energy)…  "
          "close the window to continue.")
    ani = animate_combined(sim, n_frames=args.frames,
                           sweeps_per_frame=args.sweeps_per_frame)

    plot_structure_factor(sim)


if __name__ == "__main__":
    main()
