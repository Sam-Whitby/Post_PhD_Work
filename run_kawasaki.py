"""
Run a Kawasaki lattice-gas simulation and visualise the results.

Usage
-----
    python run_kawasaki.py

Produces:
  - A combined animation: lattice image on the left, live energy plot on the right
  - A static plot of the structure factor after the run
"""

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

    # Give the energy axis a small initial y-range so it doesn't collapse
    e0 = energy_history[0]
    ax_en.set_ylim(e0 * 1.1 if e0 < 0 else e0 * 0.9,
                   e0 * 0.9 if e0 < 0 else e0 * 1.1)
    ax_en.set_xlim(0, n_frames * sweeps_per_frame)

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
        ax_en.relim()
        ax_en.autoscale_view()

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
    print(f"Initialising {L}×{L} lattice gas  "
          f"(density={DENSITY}, J={J}, T={T}, init=circle)")

    sim = KawasakiLatticeGas(L=L, density=DENSITY, J=J, T=T, seed=SEED,
                             init="circle")

    print("Showing combined animation (lattice + energy)…  "
          "close the window to continue.")
    ani = animate_combined(sim)   # keep reference so GC doesn't collect it

    plot_structure_factor(sim)


if __name__ == "__main__":
    main()
