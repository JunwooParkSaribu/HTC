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


def seg_distance(histones):
    distances = {}
    for histone in histones:
        distances[histone] = []

    for histone in histones:
        for i in range(len(histones[histone]) - 1):
            x_distance = histones[histone][i+1][0] - histones[histone][i][0]
            y_distance = histones[histone][i+1][1] - histones[histone][i][1]
            dist = np.sqrt(x_distance**2 + y_distance**2)
            distances[histone].append(dist)
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
            histone_velocity[histone].append(np.sqrt(x_distance**2 + y_distance**2)/t)
    return histone_velocity


def accumulate(histone):
    acc_histone = []
    acc = 0
    for velocity in histone:
        acc += velocity[0]
        acc_histone.append([acc])
    return acc_histone


def check_balls(histones, radius=0.2, density=0.5) -> dict:
    histones_balls = {}
    for histone in histones:
        n_balls = 0
        hybrid_flag = 0
        all_trajec = histones[histone]
        all_trajec_n = len(histones[histone])
        counted_ball = []
        for i in range(len(all_trajec)):
            if i in counted_ball:
                continue
            else:
                counted_ball = []
            trajec_density = 0
            pos = all_trajec[i][:2]
            for j in range(len(all_trajec)):
                next_pos = all_trajec[j][:2]
                if np.sqrt((next_pos[0] - pos[0])**2 + (next_pos[1] - pos[1])**2) < radius:
                    counted_ball.append(j)
                    trajec_density += 1
                else:
                    hybrid_flag = 1
            if trajec_density/all_trajec_n > density:
                n_balls += 1
            else:
                counted_ball = []
        histones_balls[histone] = [n_balls, hybrid_flag]
    del all_trajec
    return histones_balls

