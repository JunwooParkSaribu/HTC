import numpy as np


def distance(histones):
    distances = {}
    for histone in histones:
        distances[histone] = []

    for histone in histones:
        dist = 0
        for i in range(len(histones[histone]) - 1):
            x_distance = histones[histone][i+1][0] - histones[histone][i][0]
            y_distance = histones[histone][i+1][1] - histones[histone][i][1]
            dist += np.sqrt(x_distance**2 + y_distance**2)
        t = histones[histone][-1][2] - histones[histone][0][2]
        distances[histone].append(dist)
        distances[histone].append(t)
    return distances


def displacement(histones):
    displacements = {}
    for histone in histones:
        displacements[histone] = []

    for histone in histones:
        x_displacement = histones[histone][-1][0] - histones[histone][0][0]
        y_displacement = histones[histone][-1][1] - histones[histone][0][1]
        t = histones[histone][-1][2] - histones[histone][0][2]
        displacements[histone].append(np.sqrt(x_displacement ** 2 + y_displacement ** 2))
        displacements[histone].append(t)
    return displacements


def velocity(histones):
    histone_velocity = {}
    for histone in histones:
        histone_velocity[histone] = []

    for histone in histones:
        for trajec_num in range(len(histones[histone])-1):
            x_distance = histones[histone][trajec_num + 1][0] - histones[histone][trajec_num][0]
            y_distance = histones[histone][trajec_num + 1][1] - histones[histone][trajec_num][1]
            t = histones[histone][trajec_num + 1][2] - histones[histone][trajec_num][2]
            histone_velocity[histone].append([np.sqrt(x_distance**2 + y_distance**2)/t])

    return histone_velocity


def accumulate(histone):
    acc_histone = []
    acc = 0
    for velocity in histone:
        acc += velocity[0]
        acc_histone.append([acc])
    return acc_histone