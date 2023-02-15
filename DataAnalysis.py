import DataLoad
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def confusion_matrix(report):
    header, data = DataLoad.read_report(report)
    label = []
    pred = []
    for dt in data:
        label.append(dt['labeled_class_id'])
        pred.append(dt['predicted_class_id'])
    cm = metrics.confusion_matrix(label, pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                display_labels=['immobile', 'hybrid', 'mobile'])
    cm_display.plot()
    plt.show()


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


def cell_class_map(report, show=[0, 1, 2]):
    header, data = DataLoad.read_report(report)
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor("black")
    for histone in data:
        f_x_pos = float(histone['first_x_position'])
        f_y_pos = float(histone['first_y_position'])
        max_r = float(histone['maximum_radius'])
        label = int(histone['predicted_class_id'])
        if label in show:
            if label == 0:
                c = 'red'
            if label == 1:
                c = 'green'
            if label == 2:
                c = 'blue'
            plt.scatter(f_x_pos, f_y_pos, c=c, s=max_r*100, alpha=0.35)
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.show()


def cell_radius_map(report, show=[0, 1, 2]):
    header, data = DataLoad.read_report(report)
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor("black")
    mmr = 0
    x = []
    y = []
    c = []
    for histone in data:
        max_r = float(histone['maximum_radius'])
        mmr = max(mmr, max_r)
    for histone in data:
        label = int(histone['predicted_class_id'])
        if label in show:
            f_x_pos = float(histone['first_x_position'])
            f_y_pos = float(histone['first_y_position'])
            max_r = float(histone['maximum_radius'])
            #max_r = max_r / mmr * 100
            x.append(f_x_pos)
            y.append(f_y_pos)
            c.append(max_r)
    plt.scatter(x, y, c=c, s=0.5, cmap='jet', alpha=0.7)
    plt.colorbar()
    plt.show()


def bootstrapping_mean(report, repeat=1000):
    header, data = DataLoad.read_report(report)
    sample_size = len(data)
    bootstrap_mean = {'0':0, '1':0, '2':0}
    class_nums = {'0':0, '1':0, '2':0}

    for rp in range(repeat):
        for i in range(sample_size):
            sample = data[int(np.random.uniform(0, sample_size-1))]
            class_nums[sample['predicted_class_id']] += 1
        for cl in class_nums:
            class_nums[cl] /= sample_size
        for bcl in bootstrap_mean:
            bootstrap_mean[bcl] += class_nums[bcl]

    for bcl in bootstrap_mean:
        bootstrap_mean[bcl] /= repeat
    print(bootstrap_mean)


bootstrapping_mean('./result/before/all.csv', repeat=10000)
bootstrapping_mean('./result/15s/all.csv', repeat=10000)
bootstrapping_mean('./result/30s/all.csv', repeat=10000)
bootstrapping_mean('./result/1min/all.csv', repeat=10000)
bootstrapping_mean('./result/2min/all.csv', repeat=10000)
