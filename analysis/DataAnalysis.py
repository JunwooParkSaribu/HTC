from fileIO import DataLoad
from physics import TrajectoryPhy
import matplotlib.pyplot as plt


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
