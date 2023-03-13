from physics import TrajectoryPhy, DataSimulation
from fileIO import DataLoad
import numpy as np


def make_label(histones, radius=0.4, density=0.5) -> []:
    histones_balls = TrajectoryPhy.check_balls(histones, radius, density)
    for histone in histones:
        if histones_balls[histone][0] == 1 and histones_balls[histone][1] == 0:
            histones[histone].set_manuel_label(0)  # immobile
        elif histones_balls[histone][0] == 0:
            histones[histone].set_manuel_label(2)  # mobile
        else:
            histones[histone].set_manuel_label(1)  # hybrid
    del histones_balls


def label_from_report(histones, report):
    header, data = DataLoad.read_report(report)
    for histone in histones:
        for dt_dic in data:
            key = f"{dt_dic['filename']}@{dt_dic['h2b_id']}"
            if histone == key:
                histones[histone].set_manuel_label(int(dt_dic['predicted_class_id']))
    del data

    labels = [int(histones[key].get_manuel_label()) for key in histones if histones[key].get_manuel_label() is not None]
    nb_class = [0] * len(set(labels))
    keys = [[] for x in range(len(set(labels)))]
    for key in histones:
        for label in set(labels):
            if histones[key].get_manuel_label() == label:
                #keys[label].append(key)
                if label != 0:  # combination of simulated immobiles
                    keys[label].append(key)
    for label in labels:
        nb_class[label] += 1
    min_nb_class = np.min(nb_class)

    temps = []
    for i in range(len(keys)):
        if i != 0:  # combination of simulated immobiles
            selected_keys = np.random.choice(len(keys[i]), min_nb_class, replace=False)
            temp = np.array(keys[i])[selected_keys]
            temps.append(temp)
            print(f'{i}:{len(temp)}', end=' ')
    temps = np.array(temps).reshape(-1)

    new_histones = {}
    for temp in temps:
        new_histones[temp] = histones[temp]
    DataSimulation.make_immobile(new_histones, nb=min_nb_class, radius=0.15, max_distance=0.12)  # combination of simulated immobiles
    return new_histones
