"""Export the three Paper warp tests from the original instrument tables."""

from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_LOG_FOLDER = REPO_ROOT / "data" / "Paper"
TEST_DIRECTION = "warp"
OUTPUT_FOLDER = REPO_ROOT / "processed_data"
SHOW_PLOT = False
SAMPLES = ("s1", "s2", "s3")
TABLE_HEADER = ["LoadPosition", "Load", "TorquePosition", "Torque"]
TABLE_UNITS = ["(mm)", "(N)", "(Degree)", "(N-m)"]


def read_test_data(path):
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.split() == TABLE_HEADER:
                break
        else:
            raise ValueError(f"Measurement table missing in {path}")
        if next(file, "").split() != TABLE_UNITS:
            raise ValueError(f"Unexpected measurement units in {path}")
        rows = []
        for line in file:
            columns = line.split()
            if columns:
                if len(columns) < 4:
                    raise ValueError(f"Incomplete measurement in {path}: {line!r}")
                rows.append([float(value) for value in columns[:4]])
    data = np.asarray(rows, dtype=float)
    if len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f"Invalid measurement table in {path}")
    return data


def average_repeated_positions(load_pos, load_force):
    positions, indices, counts = np.unique(load_pos, return_inverse=True, return_counts=True)
    forces = np.bincount(indices, weights=load_force) / counts
    return positions, forces


def process_stretch(samples):
    loading_curves = []
    for sample in samples:
        if np.any(np.diff(sample[:, 0]) < 0):
            raise ValueError("Expected monotone stretch displacement before removing duplicates")
        load_pos, load_force = average_repeated_positions(sample[:, 0], sample[:, 1])
        turns = np.flatnonzero(np.diff(load_force) <= 0)
        end = turns[0] + 1 if turns.size else len(load_pos)
        if end < 3 or load_pos[0] != 0:
            raise ValueError("Expected an initial strictly increasing stretch segment from zero")
        loading_curves.append((load_pos[:end], load_force[:end]))

    max_pos = min(load_pos[-1] for load_pos, _ in loading_curves)
    uniform_pos = np.linspace(0, max_pos, 100)
    uniform_forces = [np.interp(uniform_pos, pos, force) for pos, force in loading_curves]
    smooth_force = savgol_filter(uniform_forces, 10, 3, axis=1).mean(axis=0)
    load_pos = np.linspace(0, max_pos, 31, endpoint=False)
    load_force = np.interp(load_pos, uniform_pos, smooth_force)
    if np.any(np.diff(load_force) <= 0):
        raise ValueError("Smoothed stretch target is not strictly increasing")
    return np.stack([load_pos, load_force - load_force.min()])


def smooth_branch(load_pos, load_force, uniform_pos):
    pos, force = average_repeated_positions(load_pos, load_force)
    smooth_force = savgol_filter(force, 10, 3)
    smooth_force[[0, -1]] = force[[0, -1]]
    return np.interp(uniform_pos, pos, smooth_force)


def process_cycles(samples, column, maximum_position, num_interp_samples, cancel_tare=False):
    uniform_up = np.linspace(0, maximum_position, num_interp_samples, endpoint=False)
    uniform_down = np.linspace(maximum_position, 0, num_interp_samples, endpoint=False)
    sample_forces = []
    for sample in samples:
        load_pos, load_force = sample[:, column:column + 2].T
        # Identify full excursions, then find the recorded troughs between peaks.
        above = load_pos > maximum_position / 2
        starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
        ends = np.flatnonzero(above & ~np.r_[above[1:], False])
        if len(starts) != 5:
            raise ValueError(f"Expected five loading cycles, found {len(starts)}")
        peaks = [start + np.argmax(load_pos[start:end + 1]) for start, end in zip(starts, ends)]
        loop_forces = []
        for i, peak in enumerate(peaks):
            before = peaks[i - 1] if i else 0
            after = peaks[i + 1] if i + 1 < len(peaks) else len(load_pos) - 1
            start = peak - np.argmin(load_pos[before:peak + 1][::-1])
            end = peak + np.argmin(load_pos[peak:after + 1])
            if load_pos[peak] < 0.98 * maximum_position or max(load_pos[start], load_pos[end]) > 0.05:
                raise ValueError("Recorded cycle does not cover the expected loading range")
            loop_forces.extend([
                smooth_branch(load_pos[start:peak + 1], load_force[start:peak + 1], uniform_up),
                smooth_branch(load_pos[peak:end + 1], load_force[peak:end + 1], uniform_down),
            ])
        sample_forces.append(np.concatenate(loop_forces))

    load_pos = np.tile(np.r_[uniform_up, uniform_down], 5)
    load_force = np.mean(sample_forces, axis=0)
    if cancel_tare:
        load_force -= np.min(load_force[:5])
    return np.stack([load_pos, load_force])


def plot_results(samples, curves, output, show_plot):
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_settings = [
        ("compression", 0, "Compression", "Displacement (mm)", "Force (N)"),
        ("twist", 2, "Twist", "Angle (deg)", "Torque (N m)"),
        ("friction", 0, "Stretch", "Displacement (mm)", "Force (N)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, (test_type, column, title, x_label, y_label) in zip(axes, plot_settings):
        for name, sample in zip(SAMPLES, samples[test_type]):
            ax.plot(sample[:, column], sample[:, column + 1], label=f"Raw {name}",
                    alpha=0.65, linewidth=1)
        load_pos, load_force = curves[test_type]
        ax.plot(load_pos, load_force, color="black", linewidth=2, label="Average")
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
        if test_type == "friction":
            ax.set_xlim(0, load_pos[-1] * 1.1)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Paper warp tests")
    path = output / "paper_warp_curves.png"
    fig.savefig(path, dpi=180)
    print(path.resolve())
    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    output = OUTPUT_FOLDER / "paper_raw_processed"

    samples = {}
    for test_type in ("compression", "twist", "friction"):
        samples[test_type] = [
            read_test_data(TEST_LOG_FOLDER / TEST_DIRECTION / sample / test_type / f"{test_type}.dat")
            for sample in SAMPLES
        ]

    curves = {
        "compression": process_cycles(samples["compression"], 0, 15.0, 50),
        "twist": process_cycles(samples["twist"], 2, 5.0, 20, cancel_tare=True),
        "friction": process_stretch(samples["friction"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    for test_type, data in curves.items():
        path = output / f"paper_{TEST_DIRECTION}_{test_type}.npy"
        with path.open("xb") as file:
            np.save(file, data)
        print(f"{path.resolve()}  {data.shape}")
    plot_results(samples, curves, output, SHOW_PLOT)


if __name__ == "__main__":
    main()
