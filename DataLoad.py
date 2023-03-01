import os
import csv
import TrajectoryPhy
from H2B import H2B
from itertools import islice


def read_file(file: str, cutoff: int) -> dict:
    histones = {}
    trajectory = {}
    time = {}
    label = {}
    labeled = False

    try:
        with open(file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')

        file_name = file.strip().split('/')[-1].strip()
        for line in lines:
            temp = line.split('\t')
            if len(temp) == 5:
                labeled = True

            temp[0] = file_name + '@' + temp[0].strip()  # filename + h2b_id
            temp[1] = float(temp[1].strip())
            temp[2] = float(temp[2].strip())
            temp[3] = float(temp[3].strip())

            if labeled:
                label[temp[0]] = int(temp[4].strip())  # label
            if temp[0] in trajectory:
                trajectory[temp[0]].append([temp[1], temp[2]])
                time[temp[0]].append(temp[3])
            else:
                trajectory[temp[0]] = [[temp[1], temp[2]]]
                time[temp[0]] = [temp[3]]

        for histone in trajectory:
            if len(trajectory[histone]) >= cutoff:
                histones[histone] = H2B()
                histones[histone].set_trajectory(trajectory[histone])
                histones[histone].set_time(time[histone])
                info = histone.strip().split('@')
                histones[histone].set_id(info[-1])
                histones[histone].set_file_name('@'.join(info[:-1]))
                if labeled:
                    histones[histone].set_manuel_label(label[histone])
        del trajectory
        del time
        del label
        TrajectoryPhy.calcul_max_radius(histones)
        return histones
    except Exception as e:
        print(f"{file} read err, {e}")


def file_distrib(paths: list, cutoff=5, group_size=2000, chunk=True) -> list:
    if os.path.isdir(paths[0]):
        files = os.listdir(paths[0])
        histones = {}
        if len(files) > 0:
            for file in files:
                if 'trxyt' in file:
                    h = read_file(paths[0] + '/' + file, cutoff=cutoff)
                    histones |= h
        if not chunk:
            return [histones]
        split_histones = []
        for item in chunks(histones, group_size):
            split_histones.append(item)
        return split_histones
    else:
        nb_files = len(paths)
        if nb_files == 1:
            h = read_file(paths[0], cutoff=cutoff)
            if not chunk:
                return [h]
            split_histones = []
            for item in chunks(h, group_size):
                split_histones.append(item)
            return split_histones
        else:
            files = paths.copy()
            histones = {}
            for file in files:
                h = read_file(file, cutoff=cutoff)
                histones |= h
            if not chunk:
                return [histones]
            split_histones = []
            for item in chunks(histones, group_size):
                split_histones.append(item)
            return split_histones


def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def read_report(file):
    lines = []
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        for row in reader:
            lines.append(row)
    return header, lines
