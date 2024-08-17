from fileIO import DataLoad
from physics import TrajectoryPhy
import matplotlib.pyplot as plt
import os


def ratio_calcul(report):
    header, data = DataLoad.read_report(report)
    total = len(data)
    immobile = 0
    hybrid = 0
    mobile = 0
    for histone in data:
        if histone['predicted_class_id'] == '0':
            immobile += 1
        if histone['predicted_class_id'] == '1':
            hybrid += 1
        if histone['predicted_class_id'] == '2':
            mobile += 1
    return immobile/total, hybrid/total, mobile/total


def hist_trajectory_length(histones):
    displacements_all = []
    displacements = TrajectoryPhy.displacement(histones)
    for h2b in displacements:
        displacements_all.extend(displacements[h2b])
    print(len(displacements_all))
    plt.figure()
    plt.hist(displacements_all)
    plt.show()


def write_trxyt_file(dict):
    times = list(dict.keys())
    base_path = '/Users/junwoopark/Downloads/analysis/whole_cells'
    for time in times:
        histones = dict[time]
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
