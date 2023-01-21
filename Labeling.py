import TrajectoryPhy


def make_label(histones, radius=0.45, density=0.4) -> []:
    histones_balls = TrajectoryPhy.check_balls(histones, radius, density)
    histone_label = {}
    for histone in histones:
        if histones_balls[histone][0] == 1 and histones_balls[histone][1] == 0:
            histone_label[histone] = 0  # immobile
        elif histones_balls[histone][0] == 0:
            histone_label[histone] = 2  # mobile
        else:
            histone_label[histone] = 1  # hybrid
    del histones_balls
    return histone_label
