import TrajectoryPhy
import DataLoad


def make_label(histones, radius=0.4, density=0.5) -> []:
    histones_balls = TrajectoryPhy.check_balls(histones, radius, density)
    histone_label = {}
    for histone in histones:
        if histones_balls[histone][0] == 1 and histones_balls[histone][1] == 0:
            histone_label[histone] = 0  # immobile
            histones[histone].set_manuel_label(0)
        elif histones_balls[histone][0] == 0:
            histone_label[histone] = 2  # mobile
            histones[histone].set_manuel_label(2)
        else:
            histone_label[histone] = 1  # hybrid
            histones[histone].set_manuel_label(1)

    del histones_balls
    return histone_label


def label_from_report(histones, report):
    histone_label = {}
    header, data = DataLoad.read_report(report)
    for histone in histones:
        for dt_dic in data:
            key = f"{dt_dic['filename']}@{dt_dic['h2b_id']}"
            if histone == key:
                histone_label[histone] = int(dt_dic['predicted_class_id'])
    del data
    return histone_label
