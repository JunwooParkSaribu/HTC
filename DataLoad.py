import os
from itertools import zip_longest
from itertools import islice


def read_file(file, cutoff=10, amplif=9):
    x_min = 99999999.
    x_max = 0.
    y_min = 99999999.
    y_max = 0.
    time_max = 0.
    histones = {}
    trajectory = {}
    file_date = file.strip().split('/')[-1].split('.')[0]

    with open(file, encoding="utf-8") as f:
        input = f.read()
    lines = input.strip().split('\n')

    for line in lines:
        temp = line.split('\t')
        temp[0] = temp[0].strip()
        temp[0] = file_date + '@' + temp[0]
        temp[1] = temp[1].strip()
        temp[1] = round(float(temp[1]), amplif)
        temp[2] = temp[2].strip()
        temp[2] = round(float(temp[2]), amplif)
        temp[3] = temp[3].strip()
        temp[3] = float(temp[3])

        if temp[1] < x_min:
            x_min = temp[1]
        if temp[1] > x_max:
            x_max = temp[1]
        if temp[2] < y_min:
            y_min = temp[2]
        if temp[2] > y_max:
            y_max = temp[2]
        if temp[3] > time_max:
            time_max = temp[3]

        if temp[0] in trajectory:
            trajectory[temp[0]].append([temp[1], temp[2], temp[3]])
        else:
            trajectory[temp[0]] = [[temp[1], temp[2], temp[3]]]

    for histone in trajectory:
        if len(trajectory[histone]) > cutoff:
            histones[histone] = trajectory[histone].copy()
    del trajectory
    return histones, x_min, x_max, y_min, y_max, time_max


def read_files(path, cutoff=10, group_size=5000, amplif=9, chunk=True):
    try:
        files = os.listdir(path)
    except Exception as e:
        print(f'File load error, current path:{path}')
        print(e)
    histones = {}
    if len(files) > 0:
        for file in files:
            if file.strip().split('.')[-1] == 'trxyt':
                h, x_min, x_max, y_min, y_max, time_max = read_file(path + '/' + file, cutoff=cutoff, amplif=amplif)
                histones |= h
    if chunk == False:
        return histones

    split_histones = []
    for item in chunks(histones, group_size):
        split_histones.append(item)
    return split_histones


def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}
