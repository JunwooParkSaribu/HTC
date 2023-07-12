import os
import sys
from fileIO import DataLoad
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import scipy
root_path = os.getcwd().split('/analysis')[0]
os.chdir(root_path)
sys.path.append(root_path)


def read_classification_file(file):
    immobile_list = []
    with open(file) as f:
        header = f.readline().strip().split(',')
        class_index = header.index('predicted_class_id')
        filename_index = header.index('filename')
        h2b_id_index = header.index('h2b_id')
        lines = f.readlines()

        for line in lines:
            line = line.strip().split(',')
            if line[class_index] == '0' and 'cluster' not in line[h2b_id_index]:
                immobile_list.append(f'{line[filename_index]}@{line[h2b_id_index]}')
    return immobile_list


def dir_search(path, histones):
    immobile_dict = {}
    imm_traj_dict = {}
    for root, dirs, files in os.walk(path, topdown=False):
        hint = root.strip().split('/')[-1]
        if hint == path.split('/')[-1]:
            continue
        immobile_dict[hint] = []

        for file in files:
            if 'trxyt.csv' in file:
                immobile_dict[hint].extend(read_classification_file(f'{root}/{file}'))

    for t in immobile_dict:
        imm_traj_dict[t] = []
        for f in immobile_dict[t]:
            imm_traj_dict[t].append(histones[f])
    return imm_traj_dict


def MSD(trajectory_dict: dict, times):
    msd = dict()
    x_axis = dict()
    for time in times:
        histone_list = trajectory_dict[time]
        disps = []
        for h2b in histone_list:
            trajectory = h2b.get_trajectory()
            t_seq = h2b.get_time()
            ref_position = trajectory[0]
            ref_t = t_seq[0]
            displacement = dict()
            for pos, t in zip(trajectory, t_seq):
                displacement[np.round(t - ref_t, 5)] = np.sqrt((pos[0] - ref_position[0])**2 + (pos[1] - ref_position[1])**2) ** 2
            disps.append(displacement)
        all_possible_times = set()
        for disp in disps:
            tmp = list(disp.keys())
            for tp in tmp:
                all_possible_times.add(tp)
        all_possible_times = list(all_possible_times)
        all_possible_times.sort()
        x_axis[time] = all_possible_times.copy()
        msd[time] = dict()
        for t in all_possible_times:
            msd[time][t] = []
            for disp in disps:
                if t in disp:
                    msd[time][t].append(disp[t])
    return msd, x_axis


if __name__ == '__main__':
    data_path = f'./data/TrainingSample/all_data'
    # Must change the backslash(\) to slash(/) or double backlash(\\) on WindowsOS
    histones = DataLoad.read_files([data_path], cutoff=2, chunk=False)[0]
    path = "/Users/junwoopark/Downloads/zone_results"

    path = path.replace('\\', '/')
    # Element names of plot_list must be same as the folder names
    plot_list = ['before', '30s', '1min', '2min']
    box_colors = ['red', 'green', 'royalblue']
    FONTSIZE = 14

    # Read classification result files (ratio, diffusion coef)
    immobile_trajectory_dict = dir_search(path, histones)
    msd, x_axis = MSD(immobile_trajectory_dict, plot_list)

    plt.figure()
    for time in plot_list:
        y_vals = []
        for t in x_axis[time]:
            y_vals.append(np.mean(msd[time][t]))
        plt.plot(x_axis[time], y_vals, label=str(time), alpha=0.7)
    plt.ylabel('MSD($um^{2}$)')
    plt.xlabel('Time(sec)')
    plt.ylim(0, 0.15)
    plt.legend()
    plt.show()
