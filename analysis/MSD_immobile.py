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


def write_trxyt_file(dict):
    times = list(dict.keys())
    base_path = '/Users/junwoopark/Downloads/fabiola/immobile_files'
    for time in times:
        histones = dict[time]
        """
        for h2b in histones:
            filename = h2b.get_file_name()
            w_path = f'{base_path}/seperated_files/{time}/{filename}'
            write_or_append = None
            if os.path.isfile(w_path):
                write_or_append = 'a'
            else:
                write_or_append = 'w'
            with open(w_path, write_or_append) as f:
                trajectory = h2b.get_trajectory()
                t_series = h2b.get_time()
                id = h2b.get_id()
                for traj, t in zip(trajectory, t_series):
                    line = f'{id}\t{traj[0]}\t{traj[1]}\t{t}\n'
                    f.write(line)
                f.close()
        """
        w_path = f'{base_path}/merged_files/{time}/merged_cells.trxyt'
        with open(w_path, 'w') as f:
            for i, h2b in enumerate(histones):
                filename = h2b.get_file_name()
                trajectory = h2b.get_trajectory()
                t_series = h2b.get_time()
                id = h2b.get_id()
                for traj, t in zip(trajectory, t_series):
                    line = f'{id}\t{traj[0]}\t{traj[1]}\t{t}\t{filename}\n'
                    f.write(line)


def MSD(trajectory_dict, plotList):
    msd = dict()
    x_axis = dict()
    for time in plotList:
        histone_list = trajectory_dict[time]
        print(len(histone_list))
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
    path = "/Users/junwoopark/Downloads/fabiola/zone_results"

    path = path.replace('\\', '/')
    # Element names of plot_list must be same as the folder names
    plot_list = ['before', '30s', '1min', '2min']
    box_colors = ['red', 'green', 'royalblue']
    FONTSIZE = 14

    # Read classification result files (ratio, diffusion coef)
    immobile_trajectory_dict = dir_search(path, histones)
    #write_trxyt_file(immobile_trajectory_dict)
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
