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


def label_from_report(histones, report, equal=False):
    header, data = DataLoad.read_report(report)
    for histone in histones:
        for dt_dic in data:
            key = f"{dt_dic['filename']}@{dt_dic['h2b_id']}"
            if histone == key:
                histones[histone].set_manuel_label(int(dt_dic['predicted_class_id']))
    del data
    if equal:
        labels = [int(histones[key].get_manuel_label()) for key in histones if histones[key].get_manuel_label() is not None]
        nb_class = [0] * len(set(labels))
        keys = [[] for x in range(len(set(labels)))]
        for key in histones:
            for label in set(labels):
                if histones[key].get_manuel_label() == label:
                    if histones[key].get_manuel_label() == 0:
                        if len(histones[key].get_trajectory()) > 30:
                            keys[label].append(key)
                    else:
                        keys[label].append(key)
        for label in labels:
            nb_class[label] += 1
        min_nb_class = np.min(nb_class)

        temps = []
        for i in range(len(keys)):
            selected_keys = np.random.choice(len(keys[i]), min_nb_class, replace=False)
            temp = np.array(keys[i])[selected_keys]
            temps.append(temp)
            print(f'{i}:{len(temp)}', end=' ')
        temps = np.array(temps).reshape(-1)

        new_histones = {}
        for temp in temps:
            new_histones[temp] = histones[temp]
        return new_histones
    else:
        return histones


def label_from_reports(histones, reports, nb_labels=3):
    headers = []
    datas = []

    for report in reports:
        header, data = DataLoad.read_report(report)
        headers.append(report)
        datas.append(data)

    key_labels = {}
    for data in datas:
        for dt_dic in data:
            key = f"{dt_dic['filename']}@{dt_dic['h2b_id']}"
            label = int(dt_dic['predicted_class_id'])

            if key not in key_labels:
                key_labels[key] = [label]
            else:
                key_labels[key].append(label)

    check_same_reports = True
    for key in key_labels:
        if len(key_labels[key]) != len(reports):
            check_same_reports = False
            print('reports are not same')
            raise Exception

    same_label_histones = [[] for x in range(nb_labels)]
    for key in key_labels:
        same = 1
        first_label = key_labels[key][0]
        for label in key_labels[key]:
            if label != first_label:
                same = 0
        if same:
            same_label_histones[first_label].append(key)

    min_nb_label = np.inf
    for group in same_label_histones:
        min_nb_label = min(len(group), min_nb_label)

    new_histones = {}
    for selec in same_label_histones:
        selected_histones = np.random.choice(selec, min_nb_label, replace=False)
        for histone in selected_histones:
            h2b_copy = histones[histone].copy()
            h2b_copy.set_manuel_label(key_labels[histone][0])
            new_histones[histone] = h2b_copy

    return new_histones



def binary_labeling(histones):  # binary labeling between immobile,mobile / hybrid
    crit = {'immobile_max_dist': 0.1697, 'immobile_max_radius': 0.4, 'immobile_min_trajectory': 10}
    #criteria: immobile: 10 to 150, max_dist = 0.1697
    #        mobile : 10 to 15

    immobile_flag = 0
    mobile_flag = 0

    new_histnoes = {}
    for histone in histones:
        trajectories = histones[histone].get_trajectory()
        if len(trajectories) < crit['immobile_min_trajectory']:
            continue

        for i in range(len(trajectories) -1):
            cur_position = trajectories[i]
            next_position = trajectories[i+1]

            displacement = np.sqrt((next_position - cur_position)[0]**2
                                   + (next_position - cur_position)[1]**2)

            if displacement > crit['immobile_max_dist']:
                mobile_flag = 1

            if 5 <= i <= len(trajectories) - 5:
                min_index = i - int(crit['immobile_min_trajectory'] / 2)
                max_index = i + int(crit['immobile_min_trajectory'] / 2)

                pivot = trajectories[i]
                flag = 0
                for internal_trajectory in trajectories[min_index:max_index]:
                    radius = np.sqrt((internal_trajectory - pivot)[0]**2 + (internal_trajectory - pivot)[1]**2)
                    if radius < crit['immobile_max_radius']:
                        flag += 1
                if flag == int(crit['immobile_min_trajectory']):
                    immobile_flag = 1

        if immobile_flag == 1 and mobile_flag == 1:
            histones[histone].set_manuel_label(1)
            new_histnoes[histone] = histones[histone]
        else:
            histones[histone].set_manuel_label(0)
            new_histnoes[histone] = histones[histone]
    return new_histnoes
