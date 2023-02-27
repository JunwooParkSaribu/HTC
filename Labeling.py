import TrajectoryPhy
import DataLoad


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
