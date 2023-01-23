import os
import csv
import TrajectoryPhy
from H2B import H2B
from itertools import islice


def read_file(file, cutoff):
    histones = {}
    trajectory = {}
    time = {}
    file_date = file.strip().split('/')[-1].split('.')[0]

    with open(file, 'r', encoding="utf-8") as f:
        input = f.read()
    lines = input.strip().split('\n')

    for line in lines:
        temp = line.split('\t')
        temp[0] = temp[0].strip()
        temp[0] = file_date + '@' + temp[0]
        temp[1] = temp[1].strip()
        temp[1] = float(temp[1])
        temp[2] = temp[2].strip()
        temp[2] = float(temp[2])
        temp[3] = temp[3].strip()
        temp[3] = float(temp[3])

        if temp[0] in trajectory:
            trajectory[temp[0]].append([temp[1], temp[2]])
            time[temp[0]].append(temp[3])
        else:
            trajectory[temp[0]] = [[temp[1], temp[2]]]
            time[temp[0]] = [temp[3]]

    for histone in trajectory:
        if len(trajectory[histone]) > cutoff:
            histones[histone] = H2B()
            histones[histone].set_trajectory(trajectory[histone])
            histones[histone].set_time(time[histone])
            info = histone.strip().split('@')
            histones[histone].set_id(info[-1])
            histones[histone].set_file_name(info[0])
    del trajectory
    del time
    TrajectoryPhy.calcul_max_radius(histones)
    return histones


def read_files(path, cutoff=10, group_size=3000, chunk=True):
    try:
        files = os.listdir(path)
        histones = {}
        if len(files) > 0:
            for file in files:
                if file.strip().split('.')[-1] == 'trxyt':
                    h = read_file(path + '/' + file, cutoff=cutoff)
                    histones |= h
        if not chunk:
            return histones
        split_histones = []
        for item in chunks(histones, group_size):
            split_histones.append(item)
        return split_histones
    except Exception as e:
        print(f'File load error, current path:{path}')
        print(e)


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
