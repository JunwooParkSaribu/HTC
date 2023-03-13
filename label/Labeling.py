from physics import TrajectoryPhy
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
                keys[label].append(key)
    for label in labels:
        nb_class[label] += 1
    min_nb_class = np.min(nb_class)
    keys = np.array(keys)

    temps = []
    print(min_nb_class)
    for i in range(len(keys)):
        selected_keys = np.random.choice(len(keys[i]), min_nb_class, replace=False)
        print(len(selected_keys))
        temp = np.array(keys[i])[selected_keys]
        temps.append(temp)
    temps = np.array(temps)
    print(temps.shape)
