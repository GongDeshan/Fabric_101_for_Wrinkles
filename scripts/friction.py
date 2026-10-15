import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

base_path = str(Path(__file__).resolve().parent.parent)

# The folder saved the raw data
test_log_folder = base_path + "//data"

# The folder you want to save the plotted fig
save_fig_path = base_path + "//plots"

# The fabric you want to process
fabric_name = "59_Cotton"
# fabric_name = "60_Linen"
# fabric_name = "61_Wool"
# fabric_name = "62_pet"
# fabric_name = "63_Silk"

# Select the test index
test_idx = "001"
# test_idx = "002"

test_direction = "warp"
test_direction = "weft"

HEAD_PATH = f"{test_log_folder}//{fabric_name}//{test_idx}//head.dat"

FILE_PATH = [f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s1//friction//friction.dat",
             f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s2//friction//friction.dat",
             f"{test_log_folder}//{fabric_name}//{test_idx}//{test_direction}//s3//friction//friction.dat"]

# --------- Start: Settings For Plot ------------

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

PLOT_TYPES = ["ALL", "AVG"]
PLOT_TYPE = PLOT_TYPES[1]

SELECTED_SAMPLE = [0, 1, 2]

IS_SMOOTH_DATA: bool = True
CANCEL_TARE: bool = True

IS_PLOT_LENGEN: bool = True
PLOT_LINE_WEIGHT = 3.0

IS_LIMIT_Y: bool = False

# --------- End: Settings For Plot ------------

EFFECTIVE_HEIGHT = 50

Interested_Proerties = ["Weight(gsm)", "Thickness(mm)"]

fabric_type = fabric_name.split("_")[1]
fig_name = f"{fabric_type}_{PLOT_TYPE}_{test_direction}_stretch.jpg"
fig_save_path = os.path.join(save_fig_path, fig_name)

Area_Density = 0.0
Thickness = 0.0

sample_load_positions = []
sample_load_forces = []

sample_load_smoothed_forces = []

sample_load_uniform_pos = []
sample_load_uniform_smoothed_forces = []

with open(HEAD_PATH, "r") as f:

    for line in f:

        line_striped = line.strip()

        if len(line_striped) > 0:

            line_tokens = line.split()

            if line_tokens[0] in Interested_Proerties:

                prop_idx = Interested_Proerties.index(line_tokens[0])

                if prop_idx == 0:
                    Area_Density = eval(line_tokens[1])
                elif prop_idx == 1:
                    Thickness = eval(line_tokens[1])

print(f"Cloth Area Density: {Area_Density} gsm, Thickness: {Thickness}")

# For Smoothing
window_length = 10
polyorder = 3

for path in FILE_PATH:

    load_pos_cnt = []
    load_positions = []
    load_forces = []

    with open(path, "r") as f:

        for line in f:

            if line[:4] == "time":

                tokens = [t for t in line.strip().split(" ") if len(t)>1]

                if len(load_positions) > 0 and tokens[3] == load_positions[-1]:
                    load_forces[-1] += eval(tokens[5])
                    load_pos_cnt[-1] += 1.0
                else:
                    load_positions.append(tokens[3])
                    load_forces.append(eval(tokens[5]))
                    load_pos_cnt.append(1.0)
                
        load_force_merge_duplicated = np.array(load_forces) / np.array(load_pos_cnt)
        load_pos_merge_duplicated = np.array([eval(p) for p in load_positions])

    smoothed_force = savgol_filter(load_force_merge_duplicated, window_length, polyorder)

    sample_load_positions.append(load_pos_merge_duplicated)

    sample_load_forces.append(load_force_merge_duplicated)

    sample_load_smoothed_forces.append(smoothed_force)

    uniform_keys_up = np.linspace(0, np.max(load_pos_merge_duplicated), 100)

if PLOT_TYPE == PLOT_TYPES[0]:

    print("Plot All")

    for s in range(len(sample_load_smoothed_forces)):
        
        if s not in SELECTED_SAMPLE:
            continue
    
        uniform_keys_up = np.linspace(0, np.max(sample_load_positions[s]), 100)

        interp_func_up = interp1d(sample_load_positions[s], sample_load_smoothed_forces[s], kind='cubic', fill_value="extrapolate")

        uniform_smoothed_forces = interp_func_up(uniform_keys_up)

        sample_load_uniform_pos.append(uniform_keys_up)

        sample_load_uniform_smoothed_forces.append(uniform_smoothed_forces)

    for i in range(len(FILE_PATH)):

        if i not in SELECTED_SAMPLE:
            continue

        if IS_SMOOTH_DATA:

            maximum_loading_pos_idx = np.argmax(sample_load_uniform_smoothed_forces[i])
            maximum_loading_pos = sample_load_uniform_pos[i][maximum_loading_pos_idx]

            uniform_loading_pos = np.linspace(0, maximum_loading_pos, 31, endpoint=False)

            avg_sample_load_position_loading = sample_load_uniform_pos[i][:maximum_loading_pos_idx+1]
            avg_sample_load_force_loading = sample_load_uniform_smoothed_forces[i][:maximum_loading_pos_idx+1]

            loading_interp_func_up = interp1d(avg_sample_load_position_loading, avg_sample_load_force_loading, kind='linear')

            average_uniform_loading_force = loading_interp_func_up(uniform_loading_pos)

            if CANCEL_TARE:
                average_uniform_loading_force = average_uniform_loading_force - np.min(average_uniform_loading_force)

            if CANCEL_TARE:
                plt.plot(sample_load_uniform_pos[i], sample_load_uniform_smoothed_forces[i] - sample_load_uniform_smoothed_forces[i][0], label=f"Smoothed Sample {i}" if len(SELECTED_SAMPLE) > 1 else "Stretch Friction", linewidth=PLOT_LINE_WEIGHT)
            else:
                plt.plot(sample_load_uniform_pos[i], sample_load_uniform_smoothed_forces[i], label=f"Smoothed Sample {i}", linewidth=PLOT_LINE_WEIGHT)

            plt.plot(uniform_loading_pos, average_uniform_loading_force, label=f"Loading Sample {i}" if len(SELECTED_SAMPLE) > 1 else "Loading", linewidth=PLOT_LINE_WEIGHT)

        else:
            plt.plot(sample_load_positions[i], sample_load_forces[i], label=f"Sample {i}", linewidth=PLOT_LINE_WEIGHT)

else:

    for s in range(len(sample_load_smoothed_forces)):
    
        uniform_keys_up = np.linspace(0, np.max(sample_load_positions[s]), 100)

        interp_func_up = interp1d(sample_load_positions[s], sample_load_smoothed_forces[s], kind='cubic', fill_value="extrapolate")

        uniform_smoothed_forces = interp_func_up(uniform_keys_up)

        sample_load_uniform_pos.append(uniform_keys_up)

        sample_load_uniform_smoothed_forces.append(uniform_smoothed_forces)
  
    avg_sample_load_position = np.average(np.stack(sample_load_uniform_pos), axis=0)
    avg_sample_load_force = np.average(np.stack(sample_load_uniform_smoothed_forces), axis=0)

    plt.plot(avg_sample_load_position, avg_sample_load_force, label="All Average", linewidth=PLOT_LINE_WEIGHT)

    # The forces when the specimen is stretched
    maximum_loading_pos_idx = np.argmax(avg_sample_load_force)
    maximum_loading_pos = avg_sample_load_position[maximum_loading_pos_idx]

    uniform_loading_pos = np.linspace(0, maximum_loading_pos, 31, endpoint=False)

    avg_sample_load_position_loading = avg_sample_load_position[:maximum_loading_pos_idx+1]
    avg_sample_load_force_loading = avg_sample_load_force[:maximum_loading_pos_idx+1]

    loading_interp_func_up = interp1d(avg_sample_load_position_loading, avg_sample_load_force_loading, kind='linear')

    average_uniform_loading_force = loading_interp_func_up(uniform_loading_pos)

    average_uniform_loading_force = average_uniform_loading_force - np.min(average_uniform_loading_force)

    with open(base_path + f"/processed_data/{fabric_name}_{test_direction}_friction.npy", "wb") as f:
        np.save(f, np.stack([uniform_loading_pos[1:], average_uniform_loading_force[1:]]))

    average_uniform_loading_force = average_uniform_loading_force

    with open(base_path + f"/processed_data/{fabric_name}_{test_direction}_friction.npy", "wb") as f:
        np.save(f, np.stack([uniform_loading_pos, average_uniform_loading_force]))

    with open(base_path + f"/processed_data/{fabric_name}_{test_direction}_friction.npy", "rb") as f:
        loaded_pos_force = np.load(f)

    plt.plot(avg_sample_load_position_loading, avg_sample_load_force_loading, label="Average Loading", linewidth=PLOT_LINE_WEIGHT)
    # plt.plot(loaded_pos_force[0], loaded_pos_force[1], label="Uniform Loading")

plt.title(f"stretch ({test_direction})".capitalize(), fontsize=30, y=1.05)

plt.xlabel("Drag Position (mm)/Strain", fontsize=28)
plt.ylabel("Load Force (N)", fontsize=28)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

if IS_LIMIT_Y:
    plt.ylim(-0.1, 6)

x_ticket_target = [dis for dis in range(0, 50, 5)]

x_ticks_comb = [ f"{t}/{t/EFFECTIVE_HEIGHT:.2f}"for t in x_ticket_target]

print(x_ticks_comb)

plt.xticks(ticks=x_ticket_target, labels=x_ticks_comb, rotation=45, ha='right')

plt.tight_layout()

if IS_PLOT_LENGEN:
    plt.legend(fontsize=16)

plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")

plt.show()     