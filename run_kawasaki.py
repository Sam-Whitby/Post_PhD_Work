"""
Run a Kawasaki lattice-gas simulation and visualise the results.

Usage
-----
    python run_kawasaki.py

Produces:
  - A live animation of the lattice evolving over time
  - A plot of energy per site vs Monte Carlo sweep
  - A plot of the structure factor (detects phase separation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kawasaki_lattice_gas import KawasakiLatticeGas


# ── Simulation parameters ──────────────────────────────────────────────────────
L        = 64       # lattice size  (L x L)
DENSITY  = 0.4      # fraction of occupied sites
J        = 1.0      # coupling constant  (J > 0 → attraction → phase separation)
T        = 1.5      # temperature  (critical T ≈ 2.27 J/k_B for 2-D Ising / lattice gas)
SEED     = 42

N_EQUILIBRATION = 200   # sweeps before measuring
N_PRODUCTION    = 500   # sweeps to measure over
MEASURE_EVERY   = 5
# ──────────────────────────────────────────────────────────────────────────────


def plot_energy(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(results["sweep"], results["energy_per_site"], lw=0.8, color="steelblue")
    ax.set_xlabel("Monte Carlo sweep")
    ax.set_ylabel("Energy per site  $E/N$")
    ax.set_title("Energy relaxation")

    ax = axes[1]
    ax.plot(results["sweep"], results["acceptance"], lw=0.8, color="darkorange")
    ax.set_xlabel("Monte Carlo sweep")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Metropolis acceptance rate")

    plt.tight_layout()
    plt.savefig("energy.png", dpi=150)
    print("Saved energy.png")
    plt.show()


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


def animate(sim: KawasakiLatticeGas, n_frames: int = 80,
            sweeps_per_frame: int = 5) -> None:
    """Show the lattice evolving in real time."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    img = ax.imshow(sim.lattice, cmap="binary", vmin=0, vmax=1,
                    interpolation="nearest")
    title = ax.set_title("Sweep 0")

    def update(frame):
        for _ in range(sweeps_per_frame):
            sim.step()
        img.set_data(sim.lattice)
        title.set_text(f"Sweep {sim.sweep}")
        return img, title

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=80, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    print(f"Initialising {L}×{L} lattice gas  "
          f"(density={DENSITY}, J={J}, T={T})")

    sim = KawasakiLatticeGas(L=L, density=DENSITY, J=J, T=T, seed=SEED)

    # ── Equilibration ──────────────────────────────────────────────────
    print(f"Equilibrating for {N_EQUILIBRATION} sweeps …")
    sim.run(N_EQUILIBRATION, measure_every=N_EQUILIBRATION + 1)   # no measurements

    # ── Production run ─────────────────────────────────────────────────
    print(f"Production run: {N_PRODUCTION} sweeps (measuring every {MEASURE_EVERY}) …")
    results = sim.run(N_PRODUCTION, measure_every=MEASURE_EVERY)

    mean_E = np.mean(results["energy_per_site"])
    print(f"Mean energy per site: {mean_E:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────
    plot_energy(results)
    plot_structure_factor(sim)

    # ── Animation (shows live lattice evolution) ───────────────────────
    print("Showing lattice animation …  (close window to exit)")
    ani = animate(sim)   # keep reference so animation is not garbage-collected


if __name__ == "__main__":
    main()
