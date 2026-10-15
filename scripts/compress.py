import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import csv

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

base_path = str(Path(__file__).resolve().parent.parent)

# The folder saved the data
test_log_folder = base_path + "//data"

# The folder you want to save the plotted fig
save_fig_path = base_path + "//plots"

# The fabric you want to check out
fabric_name = "59_Cotton"
# fabric_name = "60_Linen"
# fabric_name = "61_Wool"
# fabric_name = "62_pet"
# fabric_name = "63_Silk"

fabric_type = fabric_name.split("_")[-1]

# Select the test index
test_idx = "001"
# test_idx = "002"

test_direction = "weft"
# test_direction = "warp"
test_type = "compression"
test_type = "twist"

FILE_PATHS = [f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s1//{test_type}//{test_type}.dat",
              f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s2//{test_type}//{test_type}.dat",
              f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s3//{test_type}//{test_type}.dat" ]

TARE_VALUE_PATH = [f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s1//tarevalues.csv",
                   f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s2//tarevalues.csv",
                   f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s3//tarevalues.csv"]

SAVE_PATH = base_path + f"//processed_data//{fabric_name}_{test_direction}_{test_type}.npy"

FILE_PATHS = FILE_PATHS[:]
TARE_VALUE_PATH = TARE_VALUE_PATH[:]

IS_SMOOTH_DATA: bool = True
IS_UNIFORM_SAMPLING: bool = True
CANCEL_TARE: bool = False

# ÄLL: Plot the curves of all selected samples; AVG: Plot the mean curves of all selected samples
# select the samples by SELECTED_SAMPLES (below)
PLOT_TYPE: str = "ALL"
PLOT_TYPE: str = "AVG"

# Select the samples you want to plot [used when multiple fabric samples are tested]
SELECTED_SAMPLE: list = [0, 1, 2]

# --------- Start: For Plot ------------

IS_SAVE_PLOT: bool = False

EFFECTIVE_HEIGHT = 0.05
RADIUS = 0.04

IS_PLOT_LENGEN: bool = True
PLOT_LINE_WEIGHT = 2.0

x_label_type = "displacement" # displacement; displacement_strain
x_label_type = "displacement_strain" # displacement; displacement_strain

PLOT_X_LABEL = "Compress Position (mm)" if test_type=="compression" else "Twist Degree"

if x_label_type == "displacement":
    pass
else:
    PLOT_X_LABEL += "/Strain"

PLOT_Y_LABEL = "Force (N)" if test_type=="compression" else "Torque (N.m)"
PLOT_CAP = f"{test_type} ({test_direction})"

PLOT_CAPTION_SIZE = 30

y_lower_lim = -0.03 if test_type == "twist" else -0.9
y_up_lim = 0.07 if test_type == "twist" else 1.0

IS_LIMIT_Y: bool = False

fabric_type = fabric_name.split("_")[1]
fig_name = f"{fabric_type}_{PLOT_TYPE}_{test_direction}_{test_type}.jpg"
fig_save_path = os.path.join(save_fig_path, fig_name)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# ---------- End: For Plotting ------------

pos_idx = 3 if test_type=="compression" else 7
force_idx = 5 if test_type=="compression" else 9
NUM_INTERP_SAMPLES= 50 if test_type=="compression" else 20

def read_testing_tare_values(pathes):
        
    samples_tare_keys, samples_tare_load_values, samples_tare_torque_values = [], [], []

    samples_before_loading_tare_load, samples_before_loading_tare_torque = [], []

    for path in pathes:

        tare_keys, tare_load_values, tare_torque_values = [], [], []

        with open(path, mode='r', encoding='utf-8') as file:

            csv_reader = csv.reader(file)

            for row in csv_reader:
                if len(row) > 0:
                    if len(row) == 1:
                        tare_keys.append(row[0])
                    elif len(row) == 2:
                        if row[0] == "Load":
                            tare_load_values.append(eval(row[1]))
                        elif row[0] == "Torque":
                            tare_torque_values.append(eval(row[1]))
                    elif len(row) == 3:
                        if row[0] == "Load":
                            tare_load_values.append(eval(row[1]))
                        elif row[0] == "Torque":
                            tare_torque_values.append(eval(row[1]))

        samples_tare_keys.append(tare_keys)
        samples_tare_load_values.append(tare_load_values)
        samples_tare_torque_values.append(tare_torque_values)

        samples_before_loading_tare_load.append(tare_load_values[0])
        samples_before_loading_tare_torque.append(tare_torque_values[0])

    return samples_before_loading_tare_load, samples_before_loading_tare_torque

def read_tested_samples(pathes, tare_force, pos_idx, force_idx):
     
    samples_load_position, samples_load_force = [], []

    samples_load_positions_up = []
    samples_load_positions_down = []
    samples_load_forces_up = []
    samples_load_forces_down = []

    for sample_i, path in enumerate(pathes):

        load_pos_read, load_force_read = [], []

        loops_load_positions_up = []
        loops_load_positions_down = []
        loops_load_forces_up = []
        loops_load_forces_down = []

        with open(path, "r") as f:

            for line in f:

                if line[:4] == "time":

                    tokens = [t for t in line.strip().split(" ") if len(t)>1]

                    load_pos_read.append(tokens[pos_idx])
                    load_force_read.append(eval(tokens[force_idx]))

            pos_merge_duplicated, force_merge_duplicated, force_duplicated_cnt = [], [], []

            for p, f in zip(load_pos_read, load_force_read):

                if pos_merge_duplicated and pos_merge_duplicated[-1] == p:

                    force_merge_duplicated[-1] += f

                    force_duplicated_cnt[-1] += 1.0
                
                else:

                    pos_merge_duplicated.append(p)

                    force_merge_duplicated.append(f)

                    force_duplicated_cnt.append(1.0)

            force_merged_np = np.array(force_merge_duplicated) / np.array(force_duplicated_cnt)

            posed_merged_np = np.array([eval(p_text) for p_text in pos_merge_duplicated])

            samples_load_position.append(posed_merged_np)
            samples_load_force.append(force_merged_np - tare_force[sample_i])

        load_positions_loop_split_indices = [i for i, pos in enumerate(posed_merged_np) if pos==0.0]

        load_position_loops = [posed_merged_np[load_positions_loop_split_indices[i]:split_i] for i, split_i in enumerate(load_positions_loop_split_indices[1:])]
        load_force_loops = [force_merged_np[load_positions_loop_split_indices[i]:split_i] for i, split_i in enumerate(load_positions_loop_split_indices[1:])]

        for load_pos_one_loop, load_force_one_loop in zip(load_position_loops, load_force_loops):

            max_load_idx = np.argmax(load_pos_one_loop)

            load_forces_up = load_force_one_loop[:max_load_idx]
            load_forces_down = load_force_one_loop[max_load_idx:]

            loops_load_positions_up.append(load_pos_one_loop[:max_load_idx])
            loops_load_positions_down.append(load_pos_one_loop[max_load_idx:])
            loops_load_forces_up.append(load_forces_up - tare_force[sample_i])
            loops_load_forces_down.append(load_forces_down - tare_force[sample_i])

        samples_load_positions_up.append(loops_load_positions_up)
        samples_load_positions_down.append(loops_load_positions_down)
        samples_load_forces_up.append(loops_load_forces_up)
        samples_load_forces_down.append(loops_load_forces_down)

    return samples_load_position, samples_load_force, samples_load_positions_up, samples_load_positions_down, samples_load_forces_up, samples_load_forces_down

def smooth_raw_data(samples_load_positions_up, samples_load_positions_down, samples_load_forces_up, samples_load_forces_down, num_interp=NUM_INTERP_SAMPLES):

    # For Smoothing
    window_length = 20
    polyorder = 3

    samples_uniform_pos, samples_uniform_force = [], []
    smoothed_pos, smoothed_force = [], []

    num_samples = 0

    if not len(samples_load_positions_up) == len(samples_load_positions_down) == len(samples_load_forces_up) == len(samples_load_forces_down):
        exit(0)
    else:
        num_samples = len(samples_load_positions_up)

    for i in range(num_samples):

        loops_load_considered_up = samples_load_positions_up[i]
        loops_load_considered_down = samples_load_positions_down[i]

        loops_force_considered_up = samples_load_forces_up[i]
        loops_force_considered_down = samples_load_forces_down[i]

        loops_uniform_pos, loops_uniform_force = [], []
        loops_smoothed_pos, loops_smoothed_force = [], []

        if not len(loops_load_considered_up) == len(loops_load_considered_down) == len(loops_force_considered_up) == len(loops_force_considered_down):
            exit(0)

        for j in range(len(loops_load_considered_up)):

            # Smooth
            smoothed_force_up = savgol_filter(loops_force_considered_up[j], window_length, polyorder)
            smoothed_force_down = savgol_filter(loops_force_considered_down[j], window_length, polyorder)

            smoothed_load_positions = np.concatenate([loops_force_considered_up[j], loops_force_considered_down[j]])
            smoothed_load_forces = np.concatenate([smoothed_force_up, smoothed_force_down])

            loops_smoothed_pos.append(smoothed_load_positions)
            loops_smoothed_force.append(smoothed_load_forces)

            # Uniform Sampling
            max_rotate = np.max([np.max(loops_load_considered_up[j]), np.max(loops_load_considered_down[j])])

            uniform_keys_up = np.linspace(0, max_rotate, num_interp+1)[:-1]
            uniform_keys_down = np.linspace(max_rotate, 0.0, num_interp+1)[:-1]

            interp_func_up = interp1d(loops_load_considered_up[j], smoothed_force_up, kind='linear', fill_value="extrapolate")
            interp_func_down = interp1d(loops_load_considered_down[j], smoothed_force_down, kind='linear', fill_value="extrapolate")

            uniform_values_up = interp_func_up(uniform_keys_up)
            uniform_values_down = interp_func_down(uniform_keys_down)

            uniform_load_positions = np.concatenate([uniform_keys_up, uniform_keys_down])
            uniform_load_forces = np.concatenate([uniform_values_up, uniform_values_down])

            loops_uniform_pos.append(uniform_load_positions)
            loops_uniform_force.append(uniform_load_forces)

        smoothed_pos.append(np.concatenate(loops_smoothed_pos))
        smoothed_force.append(np.concatenate(loops_smoothed_force))

        samples_uniform_pos.append(np.concatenate(loops_uniform_pos))
        samples_uniform_force.append(np.concatenate(loops_uniform_force))

    if IS_UNIFORM_SAMPLING:
        return samples_uniform_pos, samples_uniform_force
    else:
        return smoothed_pos, smoothed_force

samples_before_loading_tare_load, samples_before_loading_tare_torque = read_testing_tare_values(TARE_VALUE_PATH)

if CANCEL_TARE:
    tare_force = samples_before_loading_tare_load if test_type=="compression" else samples_before_loading_tare_torque
else:
    tare_force = [0.0, 0.0, 0.0]

samples_load_position, samples_load_force, samples_load_positions_up, samples_load_positions_down, samples_load_forces_up, samples_load_forces_down = read_tested_samples(FILE_PATHS, tare_force, pos_idx, force_idx)
samples_uniform_pos, samples_uniform_force = smooth_raw_data(samples_load_positions_up, samples_load_positions_down, samples_load_forces_up, samples_load_forces_down, num_interp=NUM_INTERP_SAMPLES)

if PLOT_TYPE == "ALL":

    for i in range(len(FILE_PATHS)):

        if i in SELECTED_SAMPLE:

            if IS_SMOOTH_DATA:
                if CANCEL_TARE:
                    plt.plot(samples_uniform_pos[i], (samples_uniform_force[i] - samples_uniform_force[i][0]), label= f"Smoothed_Sample_{i}", linewidth=PLOT_LINE_WEIGHT)
                else:
                    plt.plot(samples_uniform_pos[i], samples_uniform_force[i], label= f"Smoothed_Sample_{i}", linewidth=PLOT_LINE_WEIGHT)
            else:
                if CANCEL_TARE:
                    plt.plot(samples_load_position[i], samples_load_force[i] - samples_load_force[i][0], label= f"Sample {i}", linewidth=PLOT_LINE_WEIGHT)  
                else:
                    plt.plot(samples_load_position[i], samples_load_force[i], label= f"Sample {i}", linewidth=PLOT_LINE_WEIGHT)  
        else:
            continue

    if IS_PLOT_LENGEN:
        plt.legend(fontsize=16)
    
    plt.title(PLOT_CAP.capitalize(), fontsize=PLOT_CAPTION_SIZE, y=1.05)

    if test_type=="twist":
        x_ticket_target = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        x_ticks_comb = [ f"{t}/{RADIUS* np.deg2rad(t)/EFFECTIVE_HEIGHT:.2f}"for t in x_ticket_target]
    else:
        x_ticket_target = [dis for dis in range(0, 18, 3)]
        x_ticks_comb = [ f"{t}/{t/1000/EFFECTIVE_HEIGHT:.2f}"for t in x_ticket_target]

    plt.xticks(ticks=x_ticket_target, labels=x_ticks_comb, rotation=45, ha='right')

    plt.xlabel(PLOT_X_LABEL, fontsize=28)
    plt.ylabel(PLOT_Y_LABEL, fontsize=28)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    if IS_LIMIT_Y:
        plt.ylim(y_lower_lim, y_up_lim)

    if IS_SAVE_PLOT:
        plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")

    plt.show()

elif PLOT_TYPE == "AVG":

    avg_samples_load_pos = np.average(np.array(samples_uniform_pos), axis=0)
    avg_samples_load_force = np.average(np.array(samples_uniform_force), axis=0)

    initial_tare_force = np.min(avg_samples_load_force[:5])

    print("Initial force: ", initial_tare_force)

    if CANCEL_TARE:
        np.save(SAVE_PATH, np.stack([avg_samples_load_pos, avg_samples_load_force - initial_tare_force]))
    else:
        np.save(SAVE_PATH, np.stack([avg_samples_load_pos, avg_samples_load_force]))

    if test_type=="twist":
        x_ticket_target = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    else:
        x_ticket_target = [dis for dis in range(0, 16)]

    x_ticks_comb = [ f"{t}/{40.0 * np.deg2rad(t)/EFFECTIVE_HEIGHT:.2f}" for t in x_ticket_target]

    plt.xticks(ticks=x_ticket_target, labels=x_ticks_comb, rotation=45, ha='right')

    read_pos_force = np.load(SAVE_PATH)

    plt.title(PLOT_CAP, fontsize=PLOT_CAPTION_SIZE)

    plt.xlabel(PLOT_X_LABEL, fontsize=28)
    plt.ylabel(PLOT_Y_LABEL, fontsize=28)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.plot(read_pos_force[0], read_pos_force[1], linewidth=PLOT_LINE_WEIGHT, c="indianred" if test_direction=="weft" else "cornflowerblue")

    plt.tight_layout()

    if IS_LIMIT_Y:
        plt.ylim(y_lower_lim, y_up_lim)

    if IS_SAVE_PLOT:
        plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")

    plt.show()