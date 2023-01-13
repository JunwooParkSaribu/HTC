import TrajectoryPhy


def make_label(histones, radius=0.35, density=0.4) -> []:
    histones_balls = TrajectoryPhy.check_balls(histones, radius, density)
    histone_label = {}
    histone_max_dist = {}
    for histone in histones:
        if histones_balls[histone][0] == 1 and histones_balls[histone][1] == 0:
            histone_label[histone] = 0  # immobile
            histone_max_dist[histone] = histones_balls[histone][2]
        elif histones_balls[histone][0] == 0:
            histone_label[histone] = 2  # mobile
            histone_max_dist[histone] = histones_balls[histone][2]
        else:
            histone_label[histone] = 1  # hybrid
            histone_max_dist[histone] = histones_balls[histone][2]
    del histones_balls
    return histone_label, histone_max_dist

