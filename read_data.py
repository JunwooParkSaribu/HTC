import os


def read_file(file, cutoff=4, amplif=6):
    x_min = 9999999.
    x_max = 0.
    y_min = 9999999.
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
        temp[0] = file_date + '@' + temp[0]
        temp[1] = round(float(temp[1]), amplif)
        temp[2] = round(float(temp[2]), amplif)
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
            histones[histone] = trajectory[histone]
    del trajectory

    return histones, x_min, x_max, y_min, y_max, time_max


def read_files(path, cutoff=4, amplif=9):
    try:
        files = os.listdir(path)
    except Exception as e:
        print(f'File load error, current path:{path}')
        print(e)
    histones = {}
    if len(files) > 0:
        for file in files:
            if file.strip().split('.')[-1] == 'trxyt':
                h, x_min, x_max, y_min, y_max, time_max = read_file(path+'/'+file, cutoff=cutoff, amplif=amplif)
                histones |= h
    return histones

